import requests

def main():
    url = 'http://127.0.0.1:5000/predict_graph'
    payload = {'dataset_subdir': 'Dataset/Ego_Twitter Stanford University/twitter/twitter'}
    r = requests.post(url, json=payload, timeout=30)
    print('STATUS', r.status_code)
    try:
        print(r.json() if r.headers.get('Content-Type','').startswith('application/json') else r.text)
    except Exception:
        print(r.text)

if __name__ == '__main__':
    main()
