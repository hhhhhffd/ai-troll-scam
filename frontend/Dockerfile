FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# nginx config is mounted at runtime to /etc/nginx/conf.d/default.conf
FROM nginx:1.29-alpine
COPY --from=build /app/dist /usr/share/nginx/html
