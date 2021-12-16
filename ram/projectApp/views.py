from django.shortcuts import render
from .import chandan

def home(request):
    return render(request,'index.html')
def result(request):
    user_input_age=int(request.GET['age'])
    prediction=chandan.fake_predict( user_input_age)



    return render(request,'result.html',{'prediction':prediction})


