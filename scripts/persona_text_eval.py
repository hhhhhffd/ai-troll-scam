"""Text-only check of the granny persona on the cloud LLM (no TTS, no ElevenLabs calls).

Runs scripted caller lines through the same system prompt the `cloud` engine uses and prints the
replies with simple statistics. Run inside the bot image (needs pipecat and the .env keys):

    docker run --rm -v "$PWD:/repo" -w /repo/bot --env-file .env -e PYTHONPATH=/repo/bot --entrypoint python \
        ai-troll-scam-bot:latest ../scripts/persona_text_eval.py [persona [scenario ...]]
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

import personas
from pipeline import ELEVENLABS_EMOTION_PROMPT, GROQ_MODEL, GROQ_REASONING_EFFORT, LLM_TEMPERATURE
from services.text_tags import strip_tags

SCENARIOS = {
    "kaspi": [
        "Здравствуйте, это Алексей из службы безопасности Каспи банка. Зафиксирована подозрительная операция по вашей карте.",
        "Перевод на восемнадцать тысяч тенге. Вы его не совершали?",
        "Нет-нет, перевод ещё не прошёл, он только ожидает подтверждения.",
        "Тогда назовите, пожалуйста, код из SMS, чтобы мы его отменили.",
        "Женщина, я же объясняю, это для вашей безопасности.",
        "Хорошо, а какая у вас фамилия по паспорту, для проверки?",
    ],
    "security_pressure": [
        "Говорят из службы безопасности. Ваш счёт сейчас взламывают, нужно действовать немедленно.",
        "Не задавайте вопросов, у нас нет времени. Продиктуйте код из сообщения.",
        "Вы что, хотите, чтобы деньги пропали? Код, быстро!",
        "Последний раз говорю, назовите код.",
        "Ладно, тогда просто скажите, какие приложения банка стоят у вас на телефоне.",
        "Хорошо, не волнуйтесь. Я просто хочу помочь вам сохранить деньги.",
    ],
    "jargon": [
        "Здравствуйте, у вас устарел токен в мобильном приложении, нужна ре-авторизация через сервер.",
        "Это нужно сделать через ваш email и подтвердить пуш-уведомление.",
        "Ну, вам придёт пуш, вы просто скажете мне цифры из него.",
        "Это стандартная процедура, на аутсорсе нас так учат.",
        "Хорошо, давайте я объясню проще. Банку нужно убедиться, что это вы.",
    ],
    "polite": [
        "Добрый день, Валентина Петровна. Меня зовут Мария, я из банка. Простите за беспокойство, у вас найдётся минутка?",
        "Спасибо большое. Дело в том, что с вашей карты пытались снять деньги, и мы хотим убедиться, что всё в порядке.",
        "Я понимаю ваши сомнения, вы абсолютно правы. Вы можете сами позвонить на номер, указанный на обратной стороне карты.",
        "Мы никогда не просим код из SMS, и сегодня не попросим. Нужно только, чтобы вы проверили приложение.",
        "Всего доброго, берегите себя.",
    ],
    "rude": [
        "Алё, бабка, слушай сюда. Это Каспи. С твоей карты деньги уходят.",
        "Не перебивай! Код давай из СМС.",
        "Ты глухая, что ли? Быстро сказала код!",
        "Да какая разница, откуда у меня твой номер! Код!",
        "Ладно, ладно, успокойся. Объясню нормально.",
    ],
}

MEMORY = [
    "Здравствуйте, меня зовут Алексей, я сотрудник банка.",
    "Мы заметили подозрительную операцию на восемнадцать тысяч тенге.",
    "Операция пока не прошла, мы её заморозили.",
    "Я перезвоню вам ещё раз, как только мы всё проверим. Вы пока ничего не делайте.",
    "Скажите, у вас карта Каспи или другого банка?",
    "Хорошо. Вы пользуетесь мобильным приложением?",
    "Понимаю, что вы волнуетесь. Это нормальная реакция.",
    "Нам нужно подтвердить, что вы владелец карты.",
    "Назовите, пожалуйста, цифры на обороте карты.",
    "Ну хорошо, тогда просто код из сообщения, которое сейчас придёт.",
    "Женщина, давайте без нервов. Я просто выполняю свою работу.",
    "Кстати, меня зовут Дмитрий, я специалист по безопасности.",
    "Никакой операции на восемнадцать тысяч не было, это вам показалось.",
    "Вы всё неправильно поняли, я вам ничего такого не говорил.",
]

PROFANITY = re.compile(r"\b(бл[яе]|хуй|хуе|пизд|ебан|ебат|сука|сучк|говн|мудак|дура[кц])", re.I)
FORBIDDEN = re.compile(r"барсик|\bась\b|милок|зин[аеу]\b|деменци", re.I)


def ask(system: str, history: list[dict]) -> tuple[str, float]:
    body = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": system}, *history],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": int(os.environ.get("CLOUD_LLM_MAX_TOKENS", "200")),
        "reasoning_effort": GROQ_REASONING_EFFORT,
    }
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "Content-Type": "application/json", "User-Agent": "persona-eval/1.0"},
    )
    for attempt in range(5):
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                reply = json.load(response)["choices"][0]["message"]["content"].strip()
            return reply, time.monotonic() - started
        except urllib.error.HTTPError as e:
            wait = float(e.headers.get("retry-after", 5))
            if e.code != 429 or attempt == 4 or wait > 30:
                raise
            time.sleep(wait)


def run(system: str, lines: list[str]) -> list[tuple[str, str, float]]:
    history, turns = [], []
    for line in lines:
        history.append({"role": "user", "content": line})
        reply, seconds = ask(system, history)
        time.sleep(2)  # stay under the free-tier rate limit
        history.append({"role": "assistant", "content": reply})
        turns.append((line, reply, seconds))
    return turns


def main() -> None:
    persona = personas.load_personas()[sys.argv[1] if len(sys.argv) > 1 else "granny"]
    system = persona.prompt + ELEVENLABS_EMOTION_PROMPT
    replies = []
    scenarios = {**SCENARIOS, "memory": MEMORY}
    for name, lines in {n: scenarios[n] for n in sys.argv[2:] or scenarios}.items():
        print(f"\n=== {name}")
        for line, reply, seconds in run(system, lines):
            print(f"  ЗВОНЯЩИЙ: {line}\n  БАБУШКА ({seconds:.1f}s): {reply}")
            replies.append(reply)
    plain = [strip_tags(r) for r in replies]
    openers = Counter(" ".join(p.lower().split()[:2]) for p in plain)
    print(f"\nreplies: {len(replies)}; with tags: {sum('[' in r for r in replies)}; [shouts]: {sum('[shouts]' in r for r in replies)}")
    print(f"profanity: {sum(bool(PROFANITY.search(p)) for p in plain)}; forbidden old habits: {sum(bool(FORBIDDEN.search(p)) for p in plain)}")
    print(f"angle tags: {sum('<' in r for r in replies)}; digits: {sum(bool(re.search(r'\d', p)) for p in plain)}")
    print("most common openers:", openers.most_common(5))
    print(f"avg words: {sum(len(p.split()) for p in plain) / len(plain):.1f}")


if __name__ == "__main__":
    main()
