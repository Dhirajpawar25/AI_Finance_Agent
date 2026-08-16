import asyncio
from app.bot import get_application

async def test():
    app = get_application()
    await app.initialize()
    updates = await app.bot.get_updates(limit=20, offset=-1)
    print('Recent updates:', len(updates))
    for u in updates:
        if u.message:
            print(f'  Update {u.update_id}: chat_id={u.message.chat.id}, user_id={u.message.from_user.id}, text={u.message.text[:50] if u.message.text else "(no text)"}')

asyncio.run(test())