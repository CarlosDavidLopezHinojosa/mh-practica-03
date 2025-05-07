# Etapa 2: producción (servidor estático)
FROM nginx:alpine

WORKDIR /app
COPY web/ .

COPY web/dist /usr/share/nginx/html
COPY web/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
