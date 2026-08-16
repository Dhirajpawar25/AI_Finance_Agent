import asyncio
from app.bot import get_application

async def test():
    app = get_application()
    await app.initialize()
    me = await app.bot.get_me()
    print('Bot info:')
    print(f'  ID: {me.id}')
    print(f'  Username: @{me.username}')
    print(f'  First name: {me.first_name}')
    print(f'  Can join groups: {me.can_join_groups}')
    print(f'  Can read all group messages: {me.can_read_all_group_messages}')
    print(f'  Supports inline queries: {me.supports_inline_queries}')

asyncio.run(test())