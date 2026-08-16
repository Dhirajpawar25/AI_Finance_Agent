from app.services.ai import get_reply

def test():
    try:
        reply = get_reply('What is 2+2?', system='You are a helpful assistant.')
        print('Reply:', reply)
    except Exception as e:
        print('Error:', type(e).__name__, str(e)[:200])

test()
