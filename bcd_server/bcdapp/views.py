from rest_framework.decorators import api_view
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from .serializers import PredictionRequestSerializer
import os
from django.conf import settings
from .predictor import prediction


@api_view(['POST'])
def predict(request):
    ser = PredictionRequestSerializer(data=request.data)
    if ser.is_valid():
        ser.save()
        sound_abs_path = os.path.normpath(str(settings.BASE_DIR) + "/.." + ser.data['sound'])
        pred = prediction.predict(sound_abs_path)
        return Response({'prediction': str(pred)}, status=status.HTTP_200_OK)
    return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
