from django.db import models


class PredictionRequestModel(models.Model):
    sound = models.FileField(default=0, upload_to="bcd_server/bcdapp/reqsounds/", )
    request_at = models.DateTimeField(null=True, blank=True)
