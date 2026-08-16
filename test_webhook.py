import asyncio
from app.bot import get_application

async def test():
    app = get_application()
    await app.initialize()
    info = await app.bot.get_webhook_info()
    print('Webhook URL:', info.url)
    print('Pending update count:', info.pending_update_count)
    print('Last error:', info.last_error_message)

asyncio.run(test())