import pytest, pytest_asyncio, httpx
from fastapi import status 
from user.operate import user_login

class TestUserOperate:
    @pytest.mark.parametrize("userName, password", [("test1","123456"),("test2","567890")])
    def test_operate_user_login(self, userName, password):
        result = user_login(userName, password)
        assert result

@pytest.mark.asyncio
class TestUserRouter:

    @pytest.mark.parametrize("userName, password", [("test1","123456"),("test2","567890")])
    async def test_user_apilogin(self, http_client:httpx.AsyncClient, userName, password):
        resp:httpx.Response = await http_client.post(url="/user/apilogin", json={"userName": userName, "password": password})
        # print(resp.json)
        assert resp.status_code == status.HTTP_200_OK
        