# 🤖 VSO Amazon Bot

Bot de Telegram para seguimiento automático de precios en Amazon.es con sistema de afiliación integrado.

## 🚀 Características

- ✅ Seguimiento automático de precios Amazon.es
- 🏆 Detección de precios mínimos históricos  
- 📊 Gráficos de evolución de precios
- 🔔 Notificaciones push inteligentes
- 💰 Enlaces de afiliación integrados
- 📈 Analytics básicos incluidos

## 🛠️ Instalación

### Opción 1: Deploy en Railway (Recomendado)
1. Crear cuenta en [Railway.app](https://railway.app)
2. Crear nuevo proyecto
3. Añadir PostgreSQL database
4. Subir estos archivos
5. Configurar variables de entorno
6. ¡Listo!

### Opción 2: Local Development
```bash
pip install -r requirements.txt
cp .env.example .env
# Editar .env con tus credenciales
python main.py
```

## ⚙️ Variables de Entorno

```env
BOT_TOKEN=tu_bot_token_de_telegram
AWS_ACCESS_KEY=tu_amazon_access_key
AWS_SECRET_KEY=tu_amazon_secret_key
ASSOCIATE_TAG=tu_associate_tag
DATABASE_URL=postgresql://... (auto en Railway)
```

## 📱 Comandos del Bot

- `/start` - Mensaje de bienvenida
- `/seguir` - Añadir producto a seguimiento  
- `/lista` - Ver productos seguidos
- `/config` - Configurar notificaciones
- `/stats` - Ver estadísticas personales

## 🎯 Uso

1. Envía cualquier enlace de Amazon.es al bot
2. El bot mostrará precio actual e historial
3. Presiona "Seguir Producto" para añadirlo
4. Recibirás alertas automáticas de bajadas de precio

## 💰 Monetización

El bot incluye enlaces de afiliación de Amazon automáticamente. Cada compra generada a través del bot produce comisiones.

## 📊 Analytics

- Seguimiento de clics en enlaces
- Usuarios más activos
- Productos más seguidos
- Estimación de conversiones

## 🔧 Tecnologías

- **Backend:** Python 3.11 + aiogram
- **Base de datos:** PostgreSQL
- **Hosting:** Railway
- **APIs:** Amazon Product Advertising API 5.0
- **Gráficos:** matplotlib + seaborn

## 📈 Escalabilidad

El bot está diseñado para manejar miles de usuarios y productos simultáneamente con verificaciones automáticas cada 5-30 minutos según la prioridad del producto.

---

*Desarrollado para maximizar ingresos por afiliación Amazon.es* 🚀
