# from django.db import models
# import json

# class Template(models.Model):
#     transId = models.AutoField(primary_key=True)
#     value = models.CharField(max_length=255,unique=True)  # A field to store the template name or description like 'template 1'
#     template = models.JSONField()  # A field to store the JSON object
    
#     def __str__(self):
#         return self.value

#====================================================================

# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class QitInvoicetemplate(models.Model):
    transid = models.AutoField(db_column='transId', primary_key=True)  # Field name made lowercase.
    value = models.CharField(unique=True, max_length=255)
    template = models.JSONField()

    class Meta:
        managed = False
        db_table = 'QIT_InvoiceTemplate'