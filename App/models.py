from django.db import models

# Create your models here.

class user_tests(models.Model):
    url = models.CharField(max_length=1000, unique=True)
    result = models.CharField(max_length=200)
    date_tested = models.DateTimeField(null=True, auto_now_add=True)