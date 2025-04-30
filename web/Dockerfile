# Etapa 1: build
FROM node:18-alpine AS builder

WORKDIR /app
COPY web/ .
RUN corepack enable && corepack prepare pnpm@latest --activate
RUN pnpm install
RUN pnpm build

# Etapa 2: producción (servidor estático)
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY web/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
