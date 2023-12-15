import pytest, httpx
from httpx_ws import aconnect_ws

@pytest.mark.asyncio
class TestOnlineWebsocket:
    async def test_group_talk(self, ws_client: httpx.AsyncClient):
        async with aconnect_ws("http://test/ws/0/tester?token=xxxx", ws_client) as websocket:
            msg = await websocket.receive_text()
            assert msg == "tester Join us"
            await websocket.send_text("hello")
            msg = await websocket.receive_text()
            assert msg == "tester: hello"

