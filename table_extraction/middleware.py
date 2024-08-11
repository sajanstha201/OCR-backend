class TokenMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        
    def __call__(self, request):
        token = request.headers.get('authorization')
        
        print(token) if token else print('No token')
        response = self.get_response(request)
        return response