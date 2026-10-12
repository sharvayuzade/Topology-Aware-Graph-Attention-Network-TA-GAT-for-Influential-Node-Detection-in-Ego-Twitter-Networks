import requests

def main():
    url = 'http://127.0.0.1:5000/predict'
    payload = {'rows': [{'node': 1, 'degree': 10}, {'node': 2, 'degree': 5}]}
    r = requests.post(url, json=payload)
    print('STATUS', r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)

if __name__ == '__main__':
    main()
