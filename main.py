# HANDLERS DE MENSAJES
    
    async def handle_product_url(self, message: types.Message, state: FSMContext):
        """Maneja URL de producto en estado de espera"""
        if not message.text or not self.is_amazon_url(message.text):
            await message.answer(
                "❌ <b>URL no válida</b>\n\n"
                "Por favor, envía un enlace válido de Amazon.es\n"
                "Ejemplo: https://amazon.es/dp/B08N5WRWNW\n\n"
                "<i>Cancelar: /cancel</i>",
                parse_mode="HTML"
            )
            return
        
        # Guardar URL en estado y pedir precio objetivo
        await state.update_data(product_url=message.text)
        await state.set_state(ProductStates.waiting_for_target_price)
        
        await message.answer(
            "💰 <b>¿Cuál es tu precio objetivo para este producto?</b>\n\n"
            "Envíame el precio máximo que estarías dispuesto a pagar.\n"
            "Te notificaré cuando el producto llegue a ese precio o menos.\n\n"
            "Ejemplos: <code>50</code>, <code>99.99</code>, <code>25,50€</code>\n\n"
            "<i>Cancelar: /cancel</i>",
            parse_mode="HTML"
        )
    
    async def handle_amazon_url(self, message: types.Message):
        """Maneja URLs de Amazon enviadas directamente"""
        await self.process_amazon_url(message, message.text)
    
    async def process_amazon_url(self, message: types.Message, url: str):
        """Procesa URL de Amazon y muestra información del producto"""
        asin = self.extract_asin_from_url(url)
        
        if not asin:
            await message.answer(
                "❌ No pude extraer el código del producto de esa URL.\n"
                "Asegúrate de que sea un enlace válido de Amazon.es"
            )
            return
        
        # Mostrar mensaje de carga
        loading_msg = await message.answer("🔍 <b>Buscando producto...</b>", parse_mode="HTML")
        
        try:
            # Obtener información del producto
            product_data = await self.amazon.get_product_info(asin)
            
            if not product_data:
                await loading_msg.edit_text(
                    "❌ <b>No pude obtener información del producto</b>\n\n"
                    "Posibles causas:\n"
                    "• El producto no está disponible\n"
                    "• URL incorrecta\n"
                    "• Error temporal de Amazon\n\n"
                    "Intenta de nuevo en unos minutos.",
                    parse_mode="HTML"
                )
                return
            
            # Guardar/actualizar producto en DB
            product_id = await self.db.add_or_update_product(product_data)
            
            # Obtener historial de precios si existe
            price_history = await self.db.get_price_history(product_id, 30)
            min_price = await self.db.get_min_price(product_id)
            
            # Verificar si ya lo sigue el usuario
            user_products = await self.db.get_user_products(message.from_user.id)
            already_following = any(p['id'] == product_id for p in user_products)
            
            # Crear mensaje del producto
            await self.send_product_info(
                message.chat.id,
                product_data,
                product_id,
                price_history,
                min_price,
                already_following,
                loading_msg.message_id
            )
            
        except Exception as e:
            logger.error(f"Error procesando URL {url}: {e}")
            await loading_msg.edit_text(
                "❌ <b>Error procesando el producto</b>\n\n"
                "Inténtalo de nuevo en unos minutos.",
                parse_mode="HTML"
            )
    
    async def send_product_info(self, chat_id: int, product: Dict, product_id: int,
                               price_history: List, min_price: float, already_following: bool,
                               edit_message_id: int = None):
        """Envía información completa del producto"""
        
        # Preparar mensaje
        title = product['title']
        price = product['price']
        
        message_text = f"📱 <b>{title}</b>\n\n"
        
        if price:
            message_text += f"💰 <b>Precio actual: {price:.2f}€</b>\n"
            
            # Indicador de precio mínimo
            if min_price and abs(price - min_price) < 0.01:
                message_text += "🏆 <b>¡PRECIO MÍNIMO HISTÓRICO!</b>\n"
            elif min_price:
                diff = price - min_price
                message_text += f"📊 Mínimo histórico: {min_price:.2f}€ (+{diff:.2f}€)\n"
        else:
            message_text += "❌ <b>Precio no disponible</b>\n"
        
        # Información adicional
        if product['availability']:
            if 'Available' in product['availability'] or 'InStock' in product['availability']:
                message_text += "✅ Disponible\n"
            else:
                message_text += f"⚠️ {product['availability']}\n"
        
        if product['rating'] and product['reviews_count']:
            stars = "⭐" * int(product['rating'])
            message_text += f"{stars} {product['rating']:.1f}/5 ({product['reviews_count']:,} reseñas)\n"
        
        # Historial de precios
        if price_history:
            message_text += f"\n📈 <b>Historial ({len(price_history)} registros):</b>\n"
            
            if len(price_history) >= 2:
                trend = "📈" if price_history[-1]['price'] > price_history[-2]['price'] else "📉"
                message_text += f"{trend} Tendencia reciente\n"
        else:
            message_text += "\n📊 <i>Primer registro de precio</i>\n"
        
        # Crear botones
        keyboard = InlineKeyboardBuilder()
        
        # Botón principal
        if min_price and price and abs(price - min_price) < 0.01:
            keyboard.row(InlineKeyboardButton(
                text="🏆 COMPRAR AL MÍNIMO HISTÓRICO",
                url=product['affiliate_url']
            ))
        else:
            keyboard.row(InlineKeyboardButton(
                text="🛒 COMPRAR AHORA",
                url=product['affiliate_url']
            ))
        
        # Botones secundarios
        if already_following:
            keyboard.row(InlineKeyboardButton(
                text="✅ Ya lo sigues",
                callback_data="noop"
            ))
        else:
            keyboard.row(InlineKeyboardButton(
                text="🔔 SEGUIR PRODUCTO",
                callback_data=f"follow_confirm:{product_id}"
            ))
        
        keyboard.row(
            InlineKeyboardButton(
                text="📊 Ver Gráfico",
                callback_data=f"history:{product_id}"
            ),
            InlineKeyboardButton(
                text="🔍 Buscar Similar",
                callback_data=f"similar:{product_id}"
            )
        )
        
        # Enviar o editar mensaje
        try:
            if edit_message_id:
                await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=edit_message_id,
                    text=message_text,
                    reply_markup=keyboard.as_markup(),
                    parse_mode="HTML"
                )
                
                # Enviar imagen por separado si está disponible
                if product['image_url']:
                    await self.bot.send_photo(
                        chat_id=chat_id,
                        photo=product['image_url'],
                        reply_markup=keyboard.as_markup()
                    )
            else:
                if product['image_url']:
                    await self.bot.send_photo(
                        chat_id=chat_id,
                        photo=product['image_url'],
                        caption=message_text,
                        reply_markup=keyboard.as_markup(),
                        parse_mode="HTML"
                    )
                else:
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=message_text,
                        reply_markup=keyboard.as_markup(),
                        parse_mode="HTML"
                    )
            
            # Registrar analytics
            await self.db.log_analytics(
                chat_id, product_id, "view_product", 
                {'source': 'url_share', 'has_image': bool(product['image_url'])}
            )
            
        except Exception as e:
            logger.error(f"Error enviando info de producto: {e}")
    
    # CALLBACK HANDLERS
    
    async def cb_confirm_follow(self, callback: CallbackQuery):
        """Confirma seguimiento de producto"""
        try:
            product_id = int(callback.data.split(":")[1])
            user_id = callback.from_user.id
            
            # Verificar que el producto existe
            async with self.db.pool.acquire() as conn:
                product = await conn.fetchrow(
                    "SELECT * FROM products WHERE id = $1", product_id
                )
            
            if not product:
                await callback.answer("❌ Producto no encontrado")
                return
            
            # Añadir seguimiento
            success = await self.db.add_user_product(user_id, product_id)
            
            if success:
                await callback.answer("✅ ¡Producto añadido a tu lista!")
                
                # Actualizar botón
                keyboard = InlineKeyboardBuilder()
                keyboard.row(InlineKeyboardButton(
                    text="🛒 COMPRAR AHORA",
                    url=product['affiliate_url']
                ))
                keyboard.row(InlineKeyboardButton(
                    text="✅ Siguiendo producto",
                    callback_data="noop"
                ))
                keyboard.row(
                    InlineKeyboardButton(
                        text="📊 Ver Gráfico",
                        callback_data=f"history:{product_id}"
                    ),
                    InlineKeyboardButton(
                        text="🗑️ Dejar de Seguir",
                        callback_data=f"unfollow:{product_id}"
                    )
                )
                
                try:
                    await callback.message.edit_reply_markup(
                        reply_markup=keyboard.as_markup()
                    )
                except:
                    pass  # Ignorar si no se puede editar
                
                # Registrar analytics
                await self.db.log_analytics(user_id, product_id, "follow_product")
                
            else:
                await callback.answer("❌ Error al seguir el producto")
                
        except Exception as e:
            logger.error(f"Error en follow confirm: {e}")
            await callback.answer("❌ Error interno")
    
    async def cb_unfollow_product(self, callback: CallbackQuery):
        """Deja de seguir producto"""
        try:
            product_id = int(callback.data.split(":")[1])
            user_id = callback.from_user.id
            
            success = await self.db.remove_user_product(user_id, product_id)
            
            if success:
                await callback.answer("✅ Producto eliminado de tu lista")
                
                # Registrar analytics
                await self.db.log_analytics(user_id, product_id, "unfollow_product")
                
                # Actualizar mensaje si es posible
                try:
                    await callback.message.delete()
                except:
                    pass
            else:
                await callback.answer("❌ Error al eliminar producto")
                
        except Exception as e:
            logger.error(f"Error en unfollow: {e}")
            await callback.answer("❌ Error interno")
    
    async def cb_show_history(self, callback: CallbackQuery):
        """Muestra historial de precios con gráfico"""
        try:
            product_id = int(callback.data.split(":")[1])
            
            # Obtener historial de precios
            price_history = await self.db.get_price_history(product_id, 30)
            
            if not price_history or len(price_history) < 2:
                await callback.answer("📊 Historial insuficiente para generar gráfico")
                return
            
            # Generar gráfico
            chart_buffer = await self.generate_price_chart(product_id, price_history)
            
            if chart_buffer:
                # Obtener info del producto
                async with self.db.pool.acquire() as conn:
                    product = await conn.fetchrow(
                        "SELECT title, current_price FROM products WHERE id = $1", 
                        product_id
                    )
                
                min_price = min(p['price'] for p in price_history)
                max_price = max(p['price'] for p in price_history)
                avg_price = sum(p['price'] for p in price_history) / len(price_history)
                
                caption = f"""
📊 <b>Historial de {product['title'][:40]}...</b>

📈 <b>Estadísticas (30 días):</b>
💰 Precio actual: {product['current_price']:.2f}€
🔻 Mínimo: {min_price:.2f}€
🔺 Máximo: {max_price:.2f}€
📊 Promedio: {avg_price:.2f}€

📅 <b>Registros:</b> {len(price_history)} puntos de datos
                """
                
                await callback.message.answer_photo(
                    photo=chart_buffer,
                    caption=caption,
                    parse_mode="HTML"
                )
                
                await callback.answer()
                
                # Registrar analytics
                await self.db.log_analytics(
                    callback.from_user.id, product_id, "view_chart"
                )
            else:
                await callback.answer("❌ Error generando gráfico")
                
        except Exception as e:
            logger.error(f"Error mostrando historial: {e}")
            await callback.answer("❌ Error interno")
    
    async def generate_price_chart(self, product_id: int, price_history: List[Dict]) -> Optional[io.BytesIO]:
        """Genera gráfico de evolución de precios"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Backend para servidores sin GUI
            
            dates = [p['recorded_at'] for p in price_history]
            prices = [float(p['price']) for p in price_history]
            
            # Configurar matplotlib
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Gráfico principal
            ax.plot(dates, prices, linewidth=2.5, color='#FF9500', marker='o', markersize=4)
            ax.fill_between(dates, prices, alpha=0.3, color='#FF9500')
            
            # Líneas de referencia
            min_price = min(prices)
            max_price = max(prices)
            avg_price = sum(prices) / len(prices)
            
            ax.axhline(y=min_price, color='green', linestyle='--', alpha=0.7, label=f'Mínimo: {min_price:.2f}€')
            ax.axhline(y=avg_price, color='blue', linestyle='--', alpha=0.7, label=f'Promedio: {avg_price:.2f}€')
            
            # Styling
            ax.set_title('Evolución del Precio - Últimos 30 días', fontsize=16, fontweight='bold')
            ax.set_xlabel('Fecha', fontsize=12)
            ax.set_ylabel('Precio (€)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Formatear fechas en eje X
            fig.autofmt_xdate()
            
            # Ajustar layout
            plt.tight_layout()
            
            # Guardar en buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plt.close(fig)
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error generando gráfico: {e}")
            return None
    
    async def cb_buy_product(self, callback: CallbackQuery):
        """Registra click en enlace de compra"""
        try:
            product_id = int(callback.data.split(":")[1])
            user_id = callback.from_user.id
            
            # Registrar click
            await self.db.log_analytics(user_id, product_id, "click_buy")
            
            await callback.answer("🛒 Redirigiendo a Amazon...")
            
        except Exception as e:
            logger.error(f"Error en buy click: {e}")
            await callback.answer()
    
    async def cb_menu_navigation(self, callback: CallbackQuery):
        """Maneja navegación del menú"""
        try:
            action = callback.data.split(":")[1]
            
            if action == "search":
                await callback.message.edit_text(
                    "🔍 <b>Buscar Producto</b>\n\n"
                    "Envíame un enlace de Amazon.es para empezar a seguir el precio.\n\n"
                    "También puedes usar:\n"
                    "• /seguir - Para buscar paso a paso\n"
                    "• Pegar enlace directamente en el chat",
                    parse_mode="HTML",
                    reply_markup=await self.create_main_menu()
                )
            
            elif action == "my_products":
                # Redirigir al comando lista
                await self.cmd_my_products(callback.message)
            
            elif action == "stats":
                await self.cmd_stats(callback.message)
            
            elif action == "settings":
                await self.cmd_settings(callback.message)
            
            elif action == "help":
                await self.cmd_help(callback.message)
            
            await callback.answer()
            
        except Exception as e:
            logger.error(f"Error en navegación de menú: {e}")
            await callback.answer("❌ Error interno")
    
    async def start_bot(self):
        """Inicia el bot y todos los servicios"""
        try:
            # Inicializar base de datos
            await self.db.init_db()
            logger.info("Base de datos inicializada")
            
            # Inicializar monitor de precios
            self.monitor = PriceMonitor(self.db, self.amazon, self.bot)
            self.monitor.start_monitoring()
            logger.info("Monitor de precios iniciado")
            
            # Configurar webhook o polling
            await self.dp.start_polling(self.bot, drop_pending_updates=True)
            
        except Exception as e:
            logger.error(f"Error iniciando bot: {e}")
            raise
    
    async def stop_bot(self):
        """Para el bot y limpia recursos"""
        try:
            if self.monitor and self.monitor.scheduler.running:
                self.monitor.scheduler.shutdown()
            
            if self.db.pool:
                await self.db.pool.close()
            
            await self.bot.session.close()
            logger.info("Bot detenido correctamente")
            
        except Exception as e:
            logger.error(f"Error deteniendo bot: {e}")

# Funciones auxiliares para despliegue en Railway

async def create_app():
    """Crea la aplicación del bot"""
    bot_instance = TelegramBot()
    return bot_instance

def setup_logging():
    """Configura el sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bot.log') if os.path.exists('/app') else logging.StreamHandler()
        ]
    )

# Script principal
async def main():
    """Función principal del bot"""
    setup_logging()
    
    bot = await create_app()
    
    try:
        logger.info("🚀 Iniciando VSO Amazon Bot...")
        await bot.start_bot()
    except KeyboardInterrupt:
        logger.info("⏹️ Bot detenido por usuario")
    except Exception as e:
        logger.error(f"💥 Error crítico: {e}")
    finally:
        await bot.stop_bot()

if __name__ == "__main__":
    asyncio.run(main())import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re
import hashlib
import hmac
import base64
from urllib.parse import quote, urlencode
import xml.etree.ElementTree as ET
import aiohttp
import asyncpg
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram import F
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
import matplotlib.pyplot as plt
import io
import numpy as np
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Configuración con validación
BOT_TOKEN = os.getenv("BOT_TOKEN")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY") 
ASSOCIATE_TAG = os.getenv("ASSOCIATE_TAG")
PAAPI_HOST = os.getenv("PAAPI_HOST", "webservices.amazon.com")
DATABASE_URL = os.getenv("DATABASE_URL")

# Validación crítica de variables
if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN no encontrado en variables de entorno")
if not AWS_ACCESS_KEY:
    raise ValueError("❌ AWS_ACCESS_KEY no encontrado en variables de entorno")
if not AWS_SECRET_KEY:
    raise ValueError("❌ AWS_SECRET_KEY no encontrado en variables de entorno")
if not ASSOCIATE_TAG:
    raise ValueError("❌ ASSOCIATE_TAG no encontrado en variables de entorno")

print(f"✅ Variables cargadas correctamente:")
print(f"   - BOT_TOKEN: {'✓' if BOT_TOKEN else '✗'}")
print(f"   - AWS_ACCESS_KEY: {'✓' if AWS_ACCESS_KEY else '✗'}")
print(f"   - AWS_SECRET_KEY: {'✓' if AWS_SECRET_KEY else '✗'}")
print(f"   - ASSOCIATE_TAG: {'✓' if ASSOCIATE_TAG else '✗'}")
print(f"   - PAAPI_HOST: {PAAPI_HOST}")
print(f"   - DATABASE_URL: {'✓' if DATABASE_URL else '✗'}")

# Estados para FSM
class TelegramBot:
    """Bot principal de Telegram"""
    
    def __init__(self):
        self.bot = Bot(token=BOT_TOKEN)
        self.dp = Dispatcher(storage=MemoryStorage())
        self.db = DatabaseManager(DATABASE_URL)
        self.amazon = AmazonAPI()
        self.monitor = None
        
        # Registrar handlers
        self.setup_handlers()
    
    def setup_handlers(self):
        """Configura los handlers del bot"""
        
        # Comandos básicos
        self.dp.message(Command("start"))(self.cmd_start)
        self.dp.message(Command("help"))(self.cmd_help)
        self.dp.message(Command("seguir"))(self.cmd_follow_product)
        self.dp.message(Command("lista"))(self.cmd_my_products)
        self.dp.message(Command("config"))(self.cmd_settings)
        self.dp.message(Command("stats"))(self.cmd_stats)
        
        # Callbacks
        self.dp.callback_query(F.data.startswith("history:"))(self.cb_show_history)
        self.dp.callback_query(F.data.startswith("unfollow:"))(self.cb_unfollow_product)
        self.dp.callback_query(F.data.startswith("follow_confirm:"))(self.cb_confirm_follow)
        self.dp.callback_query(F.data.startswith("buy:"))(self.cb_buy_product)
        self.dp.callback_query(F.data.startswith("menu:"))(self.cb_menu_navigation)
        
        # Estados FSM
        self.dp.message(ProductStates.waiting_for_url)(self.handle_product_url)
        self.dp.message(ProductStates.waiting_for_target_price)(self.handle_target_price)
        
        # Mensajes con URLs de Amazon
        self.dp.message(lambda msg: self.is_amazon_url(msg.text) if msg.text else False)(self.handle_amazon_url)
    
    def extract_asin_from_url(self, url: str) -> Optional[str]:
        """Extrae ASIN de URL de Amazon"""
        patterns = [
            r'/dp/([A-Z0-9]{10})',
            r'/gp/product/([A-Z0-9]{10})',
            r'amazon\.es/.*?/([A-Z0-9]{10})',
            r'/product/([A-Z0-9]{10})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def is_amazon_url(self, text: str) -> bool:
        """Verifica si el texto contiene URL de Amazon España"""
        if not text:
            return False
        return 'amazon.es' in text.lower() and self.extract_asin_from_url(text) is not None

    async def handle_target_price(self, message: types.Message, state: FSMContext):
        """Maneja el precio objetivo ingresado por el usuario"""
        try:
            # Obtener datos del estado
            data = await state.get_data()
            product_url = data.get('product_url')
            
            if not product_url:
                await message.answer("❌ Error: URL del producto no encontrada. Inicia de nuevo con /seguir")
                await state.clear()
                return
            
            # Validar y convertir precio
            try:
                target_price = float(message.text.replace('€', '').replace(',', '.').strip())
                if target_price <= 0:
                    raise ValueError("El precio debe ser mayor a 0")
            except ValueError:
                await message.answer(
                    "❌ **Precio inválido**\n\n"
                    "Por favor ingresa un número válido (ejemplo: 99.99 o 99,99€)"
                )
                return
            
            # Extraer ASIN del URL
            asin = self.extract_asin_from_url(product_url)
            if not asin:
                await message.answer("❌ No se pudo extraer el ASIN del producto. Verifica la URL.")
                await state.clear()
                return
            
            # Obtener información del producto
            await message.answer("🔍 **Obteniendo información del producto...**", parse_mode="Markdown")
            
            try:
                product_info = await self.amazon.get_product_info(asin)
                if not product_info:
                    await message.answer("❌ No se pudo obtener información del producto")
                    await state.clear()
                    return
                
                # Guardar producto en base de datos
                product_id = await self.db.add_or_update_product(product_info)
                
                # Añadir seguimiento del usuario
                user_id = message.from_user.id
                await self.db.add_user_product(user_id, product_id, target_price)
                
                current_price = product_info.get('price', 0)
                title = product_info.get('title', 'Producto desconocido')
                
                # Crear respuesta
                status = "🔥 **¡PRECIO OBJETIVO ALCANZADO!**" if current_price and current_price <= target_price else "📊 **Seguimiento Activo**"
                
                if current_price:
                    price_diff = current_price - target_price
                    diff_text = f"(-{abs(price_diff):.2f}€)" if price_diff < 0 else f"(+{price_diff:.2f}€)"
                    
                    response = f"""
{status}

🏷️ **Producto:** {title[:80]}{'...' if len(title) > 80 else ''}

💰 **Precios:**
• Actual: {current_price:.2f}€
• Objetivo: {target_price:.2f}€
• Diferencia: {diff_text}

🔔 **Estado:** {'¡Compra ahora!' if current_price <= target_price else 'Te notificaré cuando baje'}
"""
                else:
                    response = f"""
{status}

🏷️ **Producto:** {title[:80]}{'...' if len(title) > 80 else ''}

💰 **Precio objetivo:** {target_price:.2f}€
⚠️ **Precio actual:** No disponible

🔔 **Estado:** Te notificaré cuando esté disponible y al precio objetivo
"""
                
                # Crear teclado
                keyboard = InlineKeyboardBuilder()
                keyboard.add(InlineKeyboardButton(
                    text="📈 Ver Histórico",
                    callback_data=f"history:{product_id}"
                ))
                keyboard.add(InlineKeyboardButton(
                    text="🛒 Ver Producto",
                    url=product_info.get('affiliate_url', product_url)
                ))
                keyboard.add(InlineKeyboardButton(
                    text="❌ Eliminar Seguimiento",
                    callback_data=f"unfollow:{product_id}"
                ))
                keyboard.adjust(1)
                
                await message.answer(
                    response,
                    reply_markup=keyboard.as_markup(),
                    parse_mode="Markdown"
                )
                
                # Log del seguimiento
                logger.info(f"Nuevo seguimiento creado: User {user_id}, ASIN {asin}, Target {target_price}€")
                
            except Exception as e:
                logger.error(f"Error obteniendo producto: {e}")
                await message.answer(
                    "❌ **Error al procesar el producto**\n\n"
                    "Puede ser que:\n"
                    "• La URL no sea válida\n"
                    "• El producto no esté disponible\n"
                    "• Problemas con Amazon API\n\n"
                    "Inténtalo de nuevo con /seguir",
                    parse_mode="Markdown"
                )
            
            # Limpiar estado
            await state.clear()
            
        except Exception as e:
            logger.error(f"Error en handle_target_price: {e}")
            await message.answer("❌ Error interno. Inténtalo de nuevo con /seguir")
            await state.clear()
    
    async def create_main_menu(self) -> InlineKeyboardMarkup:
        """Crea menú principal"""
        keyboard = InlineKeyboardBuilder()
        
        keyboard.row(InlineKeyboardButton(
            text="🔍 Buscar Producto",
            callback_data="menu:search"
        ))
        
        keyboard.row(InlineKeyboardButton(
            text="📋 Mis Productos",
            callback_data="menu:my_products"
        ))
        
        keyboard.row(
            InlineKeyboardButton(
                text="📊 Estadísticas",
                callback_data="menu:stats"
            ),
            InlineKeyboardButton(
                text="⚙️ Configuración",
                callback_data="menu:settings"
            )
        )
        
        keyboard.row(InlineKeyboardButton(
            text="ℹ️ Ayuda",
            callback_data="menu:help"
        ))
        
        return keyboard.as_markup()
    
    # COMANDOS PRINCIPALES
    
    async def cmd_start(self, message: types.Message):
        """Comando /start"""
        user = message.from_user
        
        # Registrar usuario en DB
        await self.db.add_user(user.id, user.username, user.first_name)
        
        welcome_message = f"""
🎉 <b>¡Hola {user.first_name}!</b>

🤖 Soy tu asistente personal para seguir precios en Amazon.es

<b>¿Qué puedo hacer?</b>
🔍 Seguir productos de Amazon automáticamente
📉 Alertarte cuando bajan los precios
🏆 Detectar precios mínimos históricos
📊 Mostrarte gráficos de evolución
💰 Ayudarte a ahorrar dinero

<b>Para empezar:</b>
• Envíame cualquier enlace de Amazon.es
• Usa el menú de abajo para navegar
• Escribe /help para ver todos los comandos

¡Comencemos a ahorrar! 💪
        """
        
        await message.answer(
            welcome_message,
            reply_markup=await self.create_main_menu(),
            parse_mode="HTML"
        )
    
    async def cmd_help(self, message: types.Message):
        """Comando /help"""
        help_text = """
<b>🤖 COMANDOS DISPONIBLES</b>

<b>📋 Gestión de Productos:</b>
/seguir - Seguir un nuevo producto
/lista - Ver productos que sigues
/buscar - Buscar productos en Amazon

<b>⚙️ Configuración:</b>
/config - Configurar notificaciones
/horario - Cambiar horario de alertas

<b>📊 Información:</b>
/stats - Ver tus estadísticas
/historial - Ver historial de un producto

<b>🚀 Uso Rápido:</b>
• Simplemente envía un enlace de Amazon.es
• Te mostraré el precio actual e historial
• Podrás seguir el producto con un botón

<b>🔔 Tipos de Alertas:</b>
🏆 Precio mínimo histórico
📉 Bajadas de precio significativas
🎯 Precio objetivo alcanzado
📦 Cambios de disponibilidad

<b>💡 Consejos:</b>
• Sigue productos caros para mejores descuentos
• Configura precios objetivo realistas
• Activa notificaciones para no perder ofertas

¿Necesitas ayuda específica? Escríbeme y te ayudo 😊
        """
        
        await message.answer(help_text, parse_mode="HTML")
    
    async def cmd_follow_product(self, message: types.Message, state: FSMContext):
        """Comando /seguir - Inicia proceso de seguimiento"""
        await state.set_state(ProductStates.waiting_for_url)
        
        await message.answer(
            "🔗 <b>Envíame el enlace del producto de Amazon.es que quieres seguir:</b>\n\n"
            "Puedes copiar el enlace desde:\n"
            "• La app de Amazon\n"
            "• El navegador web\n"
            "• Un mensaje compartido\n\n"
            "<i>Cancelar: /cancel</i>",
            parse_mode="HTML"
        )
    
    async def cmd_my_products(self, message: types.Message):
        """Comando /lista - Muestra productos seguidos"""
        user_id = message.from_user.id
        products = await self.db.get_user_products(user_id)
        
        if not products:
            keyboard = InlineKeyboardBuilder()
            keyboard.row(InlineKeyboardButton(
                text="🔍 Buscar Producto",
                callback_data="menu:search"
            ))
            
            await message.answer(
                "📭 <b>No estás siguiendo ningún producto aún</b>\n\n"
                "Para empezar:\n"
                "• Envíame un enlace de Amazon.es\n"
                "• Usa /seguir para añadir productos\n"
                "• Presiona el botón de abajo para buscar",
                reply_markup=keyboard.as_markup(),
                parse_mode="HTML"
            )
            return
        
        message_text = f"📋 <b>Tus productos seguidos ({len(products)}):</b>\n\n"
        
        for i, product in enumerate(products[:10], 1):  # Mostrar máximo 10
            price = product['current_price']
            title = product['title'][:40] + "..." if len(product['title']) > 40 else product['title']
            
            # Obtener precio mínimo
            min_price = await self.db.get_min_price(product['id'])
            price_indicator = ""
            
            if min_price and price and abs(price - min_price) < 0.01:
                price_indicator = " 🏆"
            elif product['target_price'] and price and price <= product['target_price']:
                price_indicator = " 🎯"
            
            message_text += f"{i}. <b>{title}</b>\n"
            if price:
                message_text += f"   💰 {price:.2f}€{price_indicator}\n"
            else:
                message_text += f"   💰 No disponible{price_indicator}\n"
            if product['target_price']:
                message_text += f"   🎯 Objetivo: {product['target_price']:.2f}€\n"
            message_text += "\n"
        
        if len(products) > 10:
            message_text += f"... y {len(products) - 10} productos más\n\n"
        
        # Crear botones para navegación
        keyboard = InlineKeyboardBuilder()
        
        keyboard.row(
            InlineKeyboardButton(
                text="📊 Ver Detalles",
                callback_data="menu:product_details"
            ),
            InlineKeyboardButton(
                text="🔍 Añadir Producto",
                callback_data="menu:search"
            )
        )
        
        await message.answer(
            message_text,
            reply_markup=keyboard.as_markup(),
            parse_mode="HTML"
        )
    
    async def cmd_settings(self, message: types.Message):
        """Comando /config - Configuración de usuario"""
        keyboard = InlineKeyboardBuilder()
        
        keyboard.row(InlineKeyboardButton(
            text="🔔 Notificaciones",
            callback_data="settings:notifications"
        ))
        
        keyboard.row(InlineKeyboardButton(
            text="⏰ Horario de Alertas",
            callback_data="settings:schedule"
        ))
        
        keyboard.row(InlineKeyboardButton(
            text="💰 Precio Mínimo",
            callback_data="settings:min_price"
        ))
        
        await message.answer(
            "⚙️ <b>Configuración</b>\n\n"
            "Personaliza cómo quieres recibir las alertas:",
            reply_markup=keyboard.as_markup(),
            parse_mode="HTML"
        )
    
    async def cmd_stats(self, message: types.Message):
        """Comando /stats - Estadísticas del usuario"""
        user_id = message.from_user.id
        
        # Obtener estadísticas básicas
        products = await self.db.get_user_products(user_id)
        
        async with self.db.pool.acquire() as conn:
            # Clics totales
            total_clicks = await conn.fetchval(
                "SELECT COUNT(*) FROM analytics WHERE user_id = $1 AND action_type LIKE 'click%'",
                user_id
            ) or 0
            
            # Alertas recibidas
            total_alerts = await conn.fetchval(
                "SELECT COUNT(*) FROM analytics WHERE user_id = $1 AND action_type LIKE 'alert%'",
                user_id
            ) or 0
            
            # Registro del usuario
            user_info = await conn.fetchrow(
                "SELECT created_at, last_active FROM users WHERE user_id = $1",
                user_id
            )
        
        # Calcular ahorro estimado
        total_savings = 0
        for product in products:
            min_price = await self.db.get_min_price(product['id'])
            if min_price and product['current_price']:
                # Estimar ahorro si compró al mínimo vs precio promedio
                price_history = await self.db.get_price_history(product['id'], 30)
                if price_history:
                    avg_price = sum(p['price'] for p in price_history) / len(price_history)
                    potential_saving = avg_price - min_price
                    if potential_saving > 0:
                        total_savings += potential_saving
        
        days_using = (datetime.now() - user_info['created_at']).days if user_info else 0
        
        stats_message = f"""
📊 <b>TUS ESTADÍSTICAS</b>

📱 <b>Productos seguidos:</b> {len(products)}
🔔 <b>Alertas recibidas:</b> {total_alerts}
👆 <b>Enlaces visitados:</b> {total_clicks}
💰 <b>Ahorro estimado:</b> {total_savings:.2f}€

📅 <b>Miembro desde:</b> {user_info['created_at'].strftime('%d/%m/%Y') if user_info else 'Hoy'}
⏱️ <b>Días usando el bot:</b> {days_using}

<b>🏆 PRODUCTOS CON MEJOR RENDIMIENTO:</b>
        """
        
        # Añadir top productos por alertas
        for product in products[:3]:
            min_price = await self.db.get_min_price(product['id'])
            current = product['current_price']
            if min_price and current:
                savings = current - min_price
                if savings > 0:
                    title = product['title'][:30] + "..."
                    stats_message += f"\n💎 {title}\n   Ahorro máximo: {savings:.2f}€"
        
        await message.answer(stats_message, parse_mode="HTML") ProductStates(StatesGroup):
    waiting_for_url = State()
    waiting_for_target_price = State()

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonAPI:
    """Cliente para Amazon Product Advertising API 5.0"""
    
    def __init__(self):
        self.access_key = AWS_ACCESS_KEY
        self.secret_key = AWS_SECRET_KEY
        self.associate_tag = ASSOCIATE_TAG
        self.host = "webservices.amazon.es"
        self.region = "eu-west-1"
        self.service = "ProductAdvertisingAPI"
        self.endpoint = f"https://{self.host}/paapi5/getitems"
        
    def _sign_request(self, method: str, uri: str, query_string: str, payload: str, headers: dict) -> dict:
        """Genera firma AWS4-HMAC-SHA256"""
        
        def sign(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()
        
        def get_signature_key(key: str, date_stamp: str, region_name: str, service_name: str) -> bytes:
            k_date = sign(('AWS4' + key).encode('utf-8'), date_stamp)
            k_region = sign(k_date, region_name)
            k_service = sign(k_region, service_name)
            k_signing = sign(k_service, 'aws4_request')
            return k_signing
        
        # Crear timestamp
        t = datetime.utcnow()
        amz_date = t.strftime('%Y%m%dT%H%M%SZ')
        date_stamp = t.strftime('%Y%m%d')
        
        # Crear canonical request
        canonical_uri = uri
        canonical_querystring = query_string
        canonical_headers = '\n'.join([f'{k.lower()}:{v}' for k, v in sorted(headers.items())]) + '\n'
        signed_headers = ';'.join([k.lower() for k in sorted(headers.keys())])
        payload_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        
        canonical_request = f"{method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        
        # Crear string to sign
        algorithm = 'AWS4-HMAC-SHA256'
        credential_scope = f"{date_stamp}/{self.region}/{self.service}/aws4_request"
        string_to_sign = f"{algorithm}\n{amz_date}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        
        # Generar firma
        signing_key = get_signature_key(self.secret_key, date_stamp, self.region, self.service)
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        
        # Añadir headers de autorización
        authorization_header = f"{algorithm} Credential={self.access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
        headers['Authorization'] = authorization_header
        headers['X-Amz-Date'] = amz_date
        
        return headers
    
    async def get_product_info(self, asin: str) -> Optional[Dict]:
        """Obtiene información de producto desde Amazon API"""
        
        payload = {
            "ItemIds": [asin],
            "Resources": [
                "ItemInfo.Title",
                "ItemInfo.Features",
                "ItemInfo.ProductInfo",
                "Offers.Listings.Price",
                "Offers.Listings.DeliveryInfo",
                "Offers.Listings.Condition",
                "Offers.Listings.Availability",
                "Offers.Listings.MerchantInfo",
                "Offers.Summaries.HighestPrice",
                "Offers.Summaries.LowestPrice",
                "Images.Primary.Large",
                "Images.Primary.Medium",
                "CustomerReviews.Count",
                "CustomerReviews.StarRating",
                "BrowseNodeInfo.BrowseNodes"
            ],
            "PartnerTag": self.associate_tag,
            "PartnerType": "Associates",
            "Marketplace": "www.amazon.es"
        }
        
        import json
        payload_str = json.dumps(payload)
        
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Host': self.host,
            'X-Amz-Target': 'com.amazon.paapi5.v1.ProductAdvertisingAPIv1.GetItems',
            'Content-Encoding': 'amz-1.0'
        }
        
        # Firmar request
        headers = self._sign_request('POST', '/paapi5/getitems', '', payload_str, headers)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint, data=payload_str, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'ItemsResult' in data and 'Items' in data['ItemsResult']:
                            return self._parse_product_data(data['ItemsResult']['Items'][0])
                        else:
                            logger.error(f"Error en respuesta API: {data}")
                            return None
                    else:
                        logger.error(f"Error HTTP {response.status}: {await response.text()}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error llamando Amazon API: {e}")
            return None
    
    def _parse_product_data(self, item: Dict) -> Dict:
        """Parsea datos de producto desde respuesta API"""
        
        try:
            # Información básica
            title = item.get('ItemInfo', {}).get('Title', {}).get('DisplayValue', 'Producto desconocido')
            
            # Precio
            price = None
            currency = 'EUR'
            availability = 'Desconocido'
            
            if 'Offers' in item and 'Listings' in item['Offers']:
                listing = item['Offers']['Listings'][0]
                if 'Price' in listing and 'Amount' in listing['Price']:
                    price = listing['Price']['Amount'] / 100  # Amazon devuelve centavos
                    currency = listing['Price']['Currency']
                
                if 'Availability' in listing:
                    availability = listing['Availability']['Type']
            
            # Imagen
            image_url = None
            if 'Images' in item and 'Primary' in item['Images']:
                image_url = item['Images']['Primary']['Large']['URL']
            
            # Reviews
            reviews_count = 0
            rating = 0.0
            if 'CustomerReviews' in item:
                reviews_count = item['CustomerReviews'].get('Count', 0)
                if 'StarRating' in item['CustomerReviews']:
                    rating = item['CustomerReviews']['StarRating']['Value']
            
            # URL de afiliado
            detail_url = item.get('DetailPageURL', '')
            
            return {
                'asin': item['ASIN'],
                'title': title,
                'price': price,
                'currency': currency,
                'availability': availability,
                'image_url': image_url,
                'reviews_count': reviews_count,
                'rating': rating,
                'affiliate_url': detail_url,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error parseando datos de producto: {e}")
            return None

class DatabaseManager:
    """Gestor de base de datos PostgreSQL"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None
    
    async def init_db(self):
        """Inicializa conexión y crea tablas"""
        self.pool = await asyncpg.create_pool(self.db_url)
        
        async with self.pool.acquire() as conn:
            # Tabla usuarios
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    notifications_enabled BOOLEAN DEFAULT TRUE,
                    notification_hours_start INTEGER DEFAULT 8,
                    notification_hours_end INTEGER DEFAULT 23,
                    min_price_threshold DECIMAL(10,2) DEFAULT 10.00,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tabla productos
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    id SERIAL PRIMARY KEY,
                    asin TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT,
                    image_url TEXT,
                    affiliate_url TEXT,
                    current_price DECIMAL(10,2),
                    currency TEXT DEFAULT 'EUR',
                    availability TEXT,
                    rating DECIMAL(3,2),
                    reviews_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    check_priority INTEGER DEFAULT 30
                )
            ''')
            
            # Tabla seguimientos usuario-producto
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS user_products (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT REFERENCES users(user_id) ON DELETE CASCADE,
                    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
                    target_price DECIMAL(10,2),
                    alert_on_price_drop BOOLEAN DEFAULT TRUE,
                    alert_on_availability BOOLEAN DEFAULT TRUE,
                    alert_on_flash_deal BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, product_id)
                )
            ''')
            
            # Tabla historial de precios
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id SERIAL PRIMARY KEY,
                    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
                    price DECIMAL(10,2) NOT NULL,
                    currency TEXT DEFAULT 'EUR',
                    availability TEXT,
                    seller_info TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tabla alertas enviadas
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS sent_alerts (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    product_id INTEGER,
                    alert_type TEXT NOT NULL,
                    old_price DECIMAL(10,2),
                    new_price DECIMAL(10,2),
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tabla analytics
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS analytics (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    product_id INTEGER,
                    action_type TEXT NOT NULL,
                    clicked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
            ''')
            
            # Índices para optimización
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_products_asin ON products(asin)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_price_history_product_time ON price_history(product_id, recorded_at DESC)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_user_products_user ON user_products(user_id)')
    
    async def add_user(self, user_id: int, username: str = None, first_name: str = None):
        """Añade o actualiza usuario"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO users (user_id, username, first_name, last_active)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id) 
                DO UPDATE SET 
                    username = EXCLUDED.username,
                    first_name = EXCLUDED.first_name,
                    last_active = CURRENT_TIMESTAMP
            ''', user_id, username, first_name)
    
    async def add_or_update_product(self, product_data: Dict) -> int:
        """Añade o actualiza producto y devuelve ID"""
        async with self.pool.acquire() as conn:
            # Determinar prioridad basada en precio
            priority = 30  # default
            if product_data.get('price'):
                if product_data['price'] > 100:
                    priority = 10  # premium products
                elif product_data['price'] < 20:
                    priority = 30  # cheap products
                else:
                    priority = 15  # normal products
            
            result = await conn.fetchrow('''
                INSERT INTO products (asin, title, image_url, affiliate_url, current_price, 
                                    currency, availability, rating, reviews_count, check_priority, last_updated)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, CURRENT_TIMESTAMP)
                ON CONFLICT (asin) 
                DO UPDATE SET 
                    title = EXCLUDED.title,
                    current_price = EXCLUDED.current_price,
                    availability = EXCLUDED.availability,
                    rating = EXCLUDED.rating,
                    reviews_count = EXCLUDED.reviews_count,
                    check_priority = EXCLUDED.check_priority,
                    last_updated = CURRENT_TIMESTAMP
                RETURNING id
            ''', product_data['asin'], product_data['title'], product_data['image_url'],
                 product_data['affiliate_url'], product_data['price'], product_data['currency'],
                 product_data['availability'], product_data['rating'], product_data['reviews_count'],
                 priority)
            
            return result['id']
    
    async def add_user_product(self, user_id: int, product_id: int, target_price: float = None) -> bool:
        """Añade seguimiento de producto para usuario"""
        async with self.pool.acquire() as conn:
            try:
                await conn.execute('''
                    INSERT INTO user_products (user_id, product_id, target_price)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (user_id, product_id) DO NOTHING
                ''', user_id, product_id, target_price)
                return True
            except Exception as e:
                logger.error(f"Error añadiendo seguimiento: {e}")
                return False
    
    async def get_user_products(self, user_id: int) -> List[Dict]:
        """Obtiene productos seguidos por usuario"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT p.*, up.target_price, up.created_at as followed_since
                FROM products p
                JOIN user_products up ON p.id = up.product_id
                WHERE up.user_id = $1
                ORDER BY up.created_at DESC
            ''', user_id)
            
            return [dict(row) for row in rows]
    
    async def remove_user_product(self, user_id: int, product_id: int) -> bool:
        """Elimina seguimiento de producto"""
        async with self.pool.acquire() as conn:
            result = await conn.execute('''
                DELETE FROM user_products 
                WHERE user_id = $1 AND product_id = $2
            ''', user_id, product_id)
            
            return result == "DELETE 1"
    
    async def add_price_record(self, product_id: int, price: float, availability: str = None):
        """Añade registro de precio al historial"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO price_history (product_id, price, availability)
                VALUES ($1, $2, $3)
            ''', product_id, price, availability)
    
    async def get_price_history(self, product_id: int, days: int = 30) -> List[Dict]:
        """Obtiene historial de precios"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT price, recorded_at
                FROM price_history 
                WHERE product_id = $1 AND recorded_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                ORDER BY recorded_at ASC
            ''' % days, product_id)
            
            return [dict(row) for row in rows]
    
    async def get_products_to_check(self, limit: int = 100) -> List[Dict]:
        """Obtiene productos para verificar precios"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT p.*, COUNT(up.user_id) as followers
                FROM products p
                JOIN user_products up ON p.id = up.product_id
                WHERE p.last_updated <= CURRENT_TIMESTAMP - INTERVAL '1 minute' * p.check_priority
                GROUP BY p.id
                ORDER BY p.check_priority ASC, followers DESC
                LIMIT $1
            ''', limit)
            
            return [dict(row) for row in rows]
    
    async def get_min_price(self, product_id: int) -> Optional[float]:
        """Obtiene precio mínimo histórico"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow('''
                SELECT MIN(price) as min_price
                FROM price_history
                WHERE product_id = $1
            ''', product_id)
            
            return result['min_price'] if result else None
    
    async def get_users_following_product(self, product_id: int) -> List[Dict]:
        """Obtiene usuarios que siguen un producto"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT u.user_id, u.notifications_enabled, u.notification_hours_start,
                       u.notification_hours_end, u.min_price_threshold, up.target_price
                FROM users u
                JOIN user_products up ON u.user_id = up.user_id
                WHERE up.product_id = $1 AND u.notifications_enabled = TRUE
            ''', product_id)
            
            return [dict(row) for row in rows]
    
    async def log_analytics(self, user_id: int, product_id: int, action_type: str, metadata: Dict = None):
        """Registra analytics"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO analytics (user_id, product_id, action_type, metadata)
                VALUES ($1, $2, $3, $4)
            ''', user_id, product_id, action_type, metadata)

class PriceMonitor:
    """Monitor de precios automático"""
    
    def __init__(self, db: DatabaseManager, amazon: AmazonAPI, bot: Bot):
        self.db = db
        self.amazon = amazon
        self.bot = bot
        self.scheduler = AsyncIOScheduler()
    
    def start_monitoring(self):
        """Inicia el monitoreo automático"""
        # Verificar precios cada 5 minutos
        self.scheduler.add_job(
            self.check_all_products,
            'interval',
            minutes=5,
            id='price_check'
        )
        
        # Limpiar datos antiguos cada día
        self.scheduler.add_job(
            self.cleanup_old_data,
            'cron',
            hour=2,
            minute=0,
            id='cleanup'
        )
        
        self.scheduler.start()
        logger.info("Monitor de precios iniciado")
    
    async def check_all_products(self):
        """Verifica precios de todos los productos pendientes"""
        try:
            products = await self.db.get_products_to_check(50)  # Verificar 50 productos por vez
            logger.info(f"Verificando {len(products)} productos...")
            
            for product in products:
                await self.check_product_price(product)
                await asyncio.sleep(1)  # Respetar límites de API (1 req/sec)
                
        except Exception as e:
            logger.error(f"Error en verificación masiva: {e}")
    
    async def check_product_price(self, product: Dict):
        """Verifica precio de un producto específico"""
        try:
            # Obtener datos actuales de Amazon
            current_data = await self.amazon.get_product_info(product['asin'])
            
            if not current_data or current_data['price'] is None:
                logger.warning(f"No se pudo obtener precio para {product['asin']}")
                return
            
            old_price = product['current_price']
            new_price = current_data['price']
            
            # Actualizar producto en DB
            await self.db.add_or_update_product(current_data)
            
            # Añadir al historial si el precio cambió
            if abs(new_price - (old_price or 0)) > 0.01:
                await self.db.add_price_record(
                    product['id'], 
                    new_price, 
                    current_data['availability']
                )
                
                # Verificar si necesitamos enviar alertas
                await self.check_price_alerts(product['id'], old_price, new_price)
            
        except Exception as e:
            logger.error(f"Error verificando producto {product['asin']}: {e}")
    
    async def check_price_alerts(self, product_id: int, old_price: float, new_price: float):
        """Verifica si hay que enviar alertas de precio"""
        try:
            users = await self.db.get_users_following_product(product_id)
            
            if not users:
                return
            
            # Obtener información del producto
            product_info = await self.db.pool.fetchrow(
                'SELECT * FROM products WHERE id = $1', product_id
            )
            
            # Obtener precio mínimo histórico
            min_price = await self.db.get_min_price(product_id)
            
            for user in users:
                should_alert = False
                alert_type = ""
                
                # Verificar hora de notificaciones
                current_hour = datetime.now().hour
                if not (user['notification_hours_start'] <= current_hour <= user['notification_hours_end']):
                    continue
                
                # ALERTA PRECIO MÍNIMO HISTÓRICO
                if min_price and abs(new_price - min_price) < 0.01:
                    should_alert = True
                    alert_type = "minimum_price"
                
                # ALERTA BAJADA DE PRECIO
                elif old_price and new_price < old_price:
                    drop_percentage = ((old_price - new_price) / old_price) * 100
                    if drop_percentage >= 5:  # Bajada mínima 5%
                        should_alert = True
                        alert_type = "price_drop"
                
                # ALERTA PRECIO OBJETIVO
                elif user['target_price'] and new_price <= user['target_price']:
                    should_alert = True
                    alert_type = "target_price"
                
                if should_alert:
                    await self.send_price_alert(
                        user['user_id'], 
                        product_info, 
                        old_price, 
                        new_price, 
                        alert_type,
                        min_price
                    )
                    
        except Exception as e:
            logger.error(f"Error enviando alertas: {e}")
    
    async def send_price_alert(self, user_id: int, product: Dict, old_price: float, 
                             new_price: float, alert_type: str, min_price: float = None):
        """Envía alerta de precio personalizada"""
        try:
            # Crear mensaje según tipo de alerta
            if alert_type == "minimum_price":
                emoji = "🏆"
                title = "¡PRECIO MÍNIMO HISTÓRICO!"
                description = f"Este es el precio más bajo registrado para este producto."
            elif alert_type == "price_drop":
                emoji = "📉"
                drop_percentage = round(((old_price - new_price) / old_price) * 100, 1)
                title = f"¡Bajada de precio del {drop_percentage}%!"
                description = f"Precio anterior: {old_price:.2f}€"
            elif alert_type == "target_price":
                emoji = "🎯"
                title = "¡Precio objetivo alcanzado!"
                description = "El producto ha llegado al precio que querías."
            else:
                emoji = "💰"
                title = "Cambio de precio detectado"
                description = ""
            
            # Crear mensaje
            message = f"{emoji} <b>{title}</b>\n\n"
            message += f"📱 <b>{product['title'][:60]}...</b>\n"
            message += f"💵 <b>{new_price:.2f}€</b>"
            
            if old_price:
                if new_price < old_price:
                    message += f" <s>{old_price:.2f}€</s> ✅"
                else:
                    message += f" (antes {old_price:.2f}€)"
            
            message += f"\n{description}"
            
            # Crear botones
            keyboard = InlineKeyboardBuilder()
            
            # Botón principal de compra
            if alert_type == "minimum_price":
                keyboard.row(InlineKeyboardButton(
                    text="🏆 COMPRAR AL MÍNIMO HISTÓRICO",
                    url=product['affiliate_url']
                ))
            else:
                keyboard.row(InlineKeyboardButton(
                    text="🛒 COMPRAR AHORA",
                    url=product['affiliate_url']
                ))
            
            # Botones secundarios
            keyboard.row(
                InlineKeyboardButton(
                    text="📊 Ver Historial",
                    callback_data=f"history:{product['id']}"
                ),
                InlineKeyboardButton(
                    text="🗑️ Dejar de Seguir",
                    callback_data=f"unfollow:{product['id']}"
                )
            )
            
            # Enviar mensaje
            await self.bot.send_photo(
                chat_id=user_id,
                photo=product['image_url'] or "https://via.placeholder.com/300x300?text=Producto",
                caption=message,
                reply_markup=keyboard.as_markup(),
                parse_mode="HTML"
            )
            
            # Registrar analytics
            await self.db.log_analytics(user_id, product['id'], f"alert_{alert_type}", {
                'old_price': old_price,
                'new_price': new_price,
                'is_minimum': alert_type == "minimum_price"
            })
            
        except Exception as e:
            logger.error(f"Error enviando alerta a usuario {user_id}: {e}")
    
    async def cleanup_old_data(self):
        """Limpia datos antiguos de la base de datos"""
        try:
            async with self.db.pool.acquire() as conn:
                # Eliminar historial de precios más antiguo de 90 días
                await conn.execute('''
                    DELETE FROM price_history 
                    WHERE recorded_at < CURRENT_TIMESTAMP - INTERVAL '90 days'
                ''')
                
                # Eliminar alertas enviadas más antiguas de 30 días
                await conn.execute('''
                    DELETE FROM sent_alerts 
                    WHERE sent_at < CURRENT_TIMESTAMP - INTERVAL '30 days'
                ''')
                
                logger.info("Limpieza de datos completada")
                
        except Exception as e:
            logger.error(f"Error en limpieza de datos: {e}")

class
