import os
import ntpath
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictionRequestSerializer
from django.conf import settings
from shutil import copy
from .predictor import prediction


@api_view(['POST'])
def predict(request):
    try:
        ser = PredictionRequestSerializer(data=request.data)
        if ser.is_valid():
            ser.save()
            sound_abs_path = os.path.normpath(str(settings.BASE_DIR) + "/.." + ser.data['sound'])

            if 'type' in request.data:
                if request.data['type'] == 'cry':
                    dest_path = os.path.normpath(
                        str(settings.BASE_DIR) + "/bcdapp/reqsounds/cries/" +
                        ntpath.basename(sound_abs_path))
                    copy(sound_abs_path, dest_path)

                elif request.data['type'] == 'nope':
                    dest_path = os.path.normpath(
                        str(settings.BASE_DIR) + "/bcdapp/reqsounds/nopes/" +
                        ntpath.basename(sound_abs_path))
                    copy(sound_abs_path, dest_path)

            pred_dict = prediction.predict(sound_abs_path)
            return Response(
                {'result': pred_dict}
                , status=status.HTTP_200_OK
            )
        return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST', 'GET'])
def hello(request):
    message = "Hello dear friend"
    if 'name' in request.data:
        message = "hello dear " + request.data['name']
    return Response({"message": message}, status=status.HTTP_200_OK)
