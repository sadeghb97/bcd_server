from rest_framework import serializers
from django.utils import timezone
from .models import PredictionRequestModel


class PredictionRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionRequestModel
        fields = "__all__"
        read_only_fields = ("request_at", )

    def create(self, validated_data):
        obj = super().create(validated_data)
        obj.request_at = timezone.now()
        obj.save()
        return obj

