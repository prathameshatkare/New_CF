import jwt
import datetime

secret = "your_secret_key"

payload = {
    "userId": "U123",
    "email": "test@mail.com",
    "terminalId": "T01",
    "followingData": ["userA", "userB"],
    "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
}

token = jwt.encode(payload, secret, algorithm="HS256")
print(token)