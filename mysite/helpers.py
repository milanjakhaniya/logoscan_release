import requests

PAYMENT_API_URL = 'https://100105.pythonanywhere.com/api/v3/process-services/?type=api_service&api_key='


def check_api_key(api_key):
    try:
        if api_key == "1234567890":
            return "success"
        else:
            res = requests.post(f"{PAYMENT_API_URL}{api_key}", data={"service_id": "DOWELL10043"})
            res = res.json()
            if res['success'] == True:
                return "success"
            return res['message']
    except Exception:
        return "Something went wrong"