from django.contrib import admin
from .models import User, crop
from django.contrib.auth.admin import UserAdmin
# from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

# Register your models here.
class MyUserAdmin(UserAdmin):
    list_display = ('phone', 'first_name', 'last_name')
    search_fields = ('phone', 'first_name', 'last_name')
    readonly_fields = ('date_joined', 'last_login')

    filter_horizontal = ()
    list_filter = ()
    fieldsets = ()
    ordering = ('phone',)


admin.site.register(User, MyUserAdmin)
admin.site.register(crop)
