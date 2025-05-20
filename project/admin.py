from django.contrib import admin
from .models import Answer, Context, Question

# Register your models here.
admin.site.register(Answer)
admin.site.register(Context)
admin.site.register(Question)