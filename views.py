from django.shortcuts import render

# Create your views here.
def deepfake(request):
    return render(request,'service-page.html')