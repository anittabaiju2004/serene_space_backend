
from rest_framework import serializers
from .models import Register
from .models import *

class RegisterSerializer(serializers.ModelSerializer):
    latitude = serializers.FloatField(required=False, allow_null=True)
    longitude = serializers.FloatField(required=False, allow_null=True)

    class Meta:
        model = Register
        fields = '__all__'



# serializers.py
from rest_framework import serializers
from .models import DepressionPrediction

class DepressionPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DepressionPrediction
        fields = '__all__'


class ADHDPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ADHDPrediction
        fields = '__all__'
