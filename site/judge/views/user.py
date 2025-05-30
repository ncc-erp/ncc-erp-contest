import base64
import itertools
import json
import os
import hmac
import hashlib
from urllib.parse import parse_qsl
from datetime import datetime
from operator import attrgetter, itemgetter
import requests
from django.contrib import messages
from django.conf import settings
from django.contrib.auth import logout as auth_logout, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.models import Permission, User
from django.contrib.auth.views import LoginView, PasswordChangeView, PasswordResetView, redirect_to_login
from django.contrib.contenttypes.models import ContentType
from django.core.cache import cache
from django.core.exceptions import PermissionDenied, ValidationError
from django.db.models import Count, Max, Min
from django.db.models.functions import ExtractYear, TruncDate
from django.http import Http404, HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import get_object_or_404, render, redirect
from django.urls import reverse
from django.utils import timezone
from django.utils.formats import date_format
from django.utils.functional import cached_property
from django.utils.safestring import mark_safe
from django.utils.translation import gettext as _, gettext_lazy
from django.views.decorators.http import require_POST
from django.views.generic import DetailView, FormView, ListView, TemplateView, View
from reversion import revisions

from judge.forms import CustomAuthenticationForm, DownloadDataForm, ProfileForm, newsletter_id
from judge.models import Profile, Submission, Language
from judge.performance_points import get_pp_breakdown
from judge.ratings import rating_class, rating_progress
from judge.tasks import prepare_user_data
from judge.utils.celery import task_status_by_id, task_status_url_by_id
from judge.utils.infinite_paginator import InfinitePaginationMixin
from judge.utils.problems import contest_completed_ids, user_completed_ids
from judge.utils.pwned import PwnedPasswordsValidator
from judge.utils.ranker import ranker
from judge.utils.subscription import Subscription
from judge.utils.unicode import utf8text
from judge.utils.views import DiggPaginatorMixin, QueryStringSortMixin, TitleMixin, add_file_response, generic_message
from judge.views import TitledTemplateView
from .contests import ContestRanking

__all__ = ['UserPage', 'UserAboutPage', 'UserProblemsPage', 'UserDownloadData', 'UserPrepareData',
           'users', 'edit_profile']


def remap_keys(iterable, mapping):
    return [dict((mapping.get(k, k), v) for k, v in item.items()) for item in iterable]


class UserMixin(object):
    model = Profile
    slug_field = 'user__username'
    slug_url_kwarg = 'user'
    context_object_name = 'user'

    def render_to_response(self, context, **response_kwargs):
        return super(UserMixin, self).render_to_response(context, **response_kwargs)


class UserPage(TitleMixin, UserMixin, DetailView):
    template_name = 'user/user-base.html'

    def get_object(self, queryset=None):
        if self.kwargs.get(self.slug_url_kwarg, None) is None:
            return self.request.profile
        return super(UserPage, self).get_object(queryset)

    def dispatch(self, request, *args, **kwargs):
        if self.kwargs.get(self.slug_url_kwarg, None) is None:
            if not self.request.user.is_authenticated:
                return redirect_to_login(self.request.get_full_path())
        try:
            return super(UserPage, self).dispatch(request, *args, **kwargs)
        except Http404:
            return generic_message(request, _('No such user'), _('No user handle "%s".') %
                                   self.kwargs.get(self.slug_url_kwarg, None))

    def get_title(self):
        return (_('My account') if self.request.user == self.object.user else
                _('User %s') % self.object.display_name)

    # TODO: the same code exists in problem.py, maybe move to problems.py?
    @cached_property
    def profile(self):
        if not self.request.user.is_authenticated:
            return None
        return self.request.profile

    @cached_property
    def in_contest(self):
        return self.profile is not None and self.profile.current_contest is not None

    def get_completed_problems(self):
        if self.in_contest:
            return contest_completed_ids(self.profile.current_contest)
        else:
            return user_completed_ids(self.profile) if self.profile is not None else ()

    def get_context_data(self, **kwargs):
        context = super(UserPage, self).get_context_data(**kwargs)

        context['hide_solved'] = int(self.hide_solved)
        context['authored'] = self.object.authored_problems.filter(is_public=True, is_organization_private=False) \
                                  .select_related('group').order_by('code')
        rating = self.object.ratings.order_by('-contest__end_time')[:1]
        context['rating'] = rating[0] if rating else None

        context['rank'] = Profile.objects.filter(
            is_unlisted=False, performance_points__gt=self.object.performance_points,
        ).exclude(id=self.object.id).count() + 1

        if rating:
            context['rating_rank'] = Profile.objects.filter(
                is_unlisted=False, rating__gt=self.object.rating,
            ).count() + 1
        context.update(self.object.ratings.aggregate(min_rating=Min('rating'), max_rating=Max('rating'),
                                                     contests=Count('contest')))
        return context

    def get(self, request, *args, **kwargs):
        self.hide_solved = request.GET.get('hide_solved') == '1' if 'hide_solved' in request.GET else False
        return super(UserPage, self).get(request, *args, **kwargs)


class CustomLoginView(LoginView):
    template_name = 'registration/login.html'
    extra_context = {'title': gettext_lazy('Login')}
    authentication_form = CustomAuthenticationForm
    redirect_authenticated_user = True

    def form_valid(self, form):
        password = form.cleaned_data['password']
        validator = PwnedPasswordsValidator()
        try:
            validator.validate(password)
        except ValidationError:
            self.request.session['password_pwned'] = True
        else:
            self.request.session['password_pwned'] = False
        return super().form_valid(form)


class CustomLoginCallbackView(LoginView):
    template_name = 'registration/callback_login.html'
    extra_context = {'title': gettext_lazy('Mezon Authentication')}
    redirect_authenticated_user = True
    
    def get_context_data(self, **kwargs):
        # Get query parameters from the URL
        auth_code = self.request.GET.get('code', '')
        # Add query param to context
        context = super().get_context_data(**kwargs)
        context['code'] = auth_code
        return context
    
    def get(self, request, *args, **kwargs):
        auth_code = self.request.GET.get('code', '')
        
        if not auth_code:
            # Handle error, e.g., missing authorization code
            return redirect('auth_login')
        
        mezon_auth_url = getattr(settings, 'MEZON_AUTH_URL')
        mezon_client_id = getattr(settings, 'MEZON_AUTH_CLIENT_ID')
        mezon_client_secret = getattr(settings, 'MEZON_AUTH_CLIENT_SECRET')
        redirect_url = getattr(settings, 'MEZON_AUTH_REDIRECT_URL')
        
        body = {
            'client_id': mezon_client_id,
            'code': auth_code,
            'client_secret': mezon_client_secret,
            'redirect_uri': redirect_url,
            'grant_type': 'authorization_code',
        }
        auth_response = requests.post(f'{mezon_auth_url}/oauth2/token', data=body)

        if auth_response.status_code != 200:
            messages.error(request, _('Failed to authenticate with Mezon'))
            return redirect('auth_login')
        
        auth_data = auth_response.json()
        access_token = auth_data.get('access_token')
        if not access_token:
            return redirect_to_login()
        # Fetch user data using the access token
        user_data = self.fetch_user_data(access_token)
        if not user_data:
            return redirect_to_login()
            
        # Process user data and log in the user
        user_email = user_data.get('sub')
        username = user_data.get('username')  # Get username from OAuth response
        
        login_user = User.objects.filter(email=user_email).first()
        if not login_user:
            # Create new user with data from OAuth
            login_user = User.objects.create_user(
                username=username,
                email=user_email,
            )
            # Create profile for the new user
            profile = Profile(user=login_user)
            profile.language = Language.get_default_language()
            profile.save()
            
            # Log in the user
            login(request, login_user, backend='django.contrib.auth.backends.ModelBackend')  
            # Redirect to profile creation page to complete profile setup
            return redirect('profile_creation')
            
        # Log in with existing user
        login(request, login_user, backend='django.contrib.auth.backends.ModelBackend')
        return redirect('home')
        
    def fetch_user_data(self, access_token):
        # Fetch user data from the OAuth provider using the access token
        mezon_auth_url = getattr(settings, 'MEZON_AUTH_URL')
        headers = {'Authorization': f'Bearer {access_token}'}
        response = requests.get(f'{mezon_auth_url}/userinfo', headers=headers)

        if response.status_code == 200:
            return response.json()  # Assuming the user data is returned in JSON format
        else:
            raise Exception('Unable to fetch user data from the provider')


class CustomLoginHashView(LoginView):
    template_name = 'registration/callback_login.html'
    extra_context = {'title': gettext_lazy('Mezon Authentication')}
    redirect_authenticated_user = True

    def get(self, request, *args, **kwargs):
        base64_data = request.GET.get('mezon_auth', '')
        if not base64_data:
            # Handle error, e.g., missing authorization code
            return redirect_to_login()
        auth_data = base64.b64decode(base64_data).decode('utf-8')
        mezon_app_token = getattr(settings, 'MEZON_APP_TOKEN')
        secret_key = self.HMAC_SHA256(mezon_app_token, "WebAppData")
        is_valid = self.validate_hash(auth_data, secret_key)
        if not is_valid:
            return redirect_to_login()
        data_parsed = dict(parse_qsl(auth_data))
        user_data = json.loads(data_parsed.get('user'))
        # Process user data and log in the user
        user_email = user_data.get('mezon_id')
        login_user = User.objects.filter(email=user_email).first()
        if not login_user:
            messages.error(request, _('Account not exists in the system'))
            return redirect('auth_login')
        # Log in with the user
        login(request, login_user, backend='django.contrib.auth.backends.ModelBackend')
        return redirect('home')
    
    def validate_hash(self, data, key) -> bool:
        [hash_params, hash_value] = data.split('&hash=')
        hashed_data = self.HEX(self.HMAC_SHA256(key, hash_params))
        return hashed_data == hash_value
        
    def HMAC_SHA256(self, key, data) -> bytes:
        if isinstance(key, str):
            key = key.encode()
        if isinstance(data, str):
            data = data.encode()
        return hmac.new(key, data, hashlib.sha256).digest()

    def HEX(self, data: bytes) -> str:
        return data.hex()
        

class CustomPasswordChangeView(PasswordChangeView):
    template_name = 'registration/password_change_form.html'

    def form_valid(self, form):
        self.request.session['password_pwned'] = False
        return super().form_valid(form)


EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class UserAboutPage(UserPage):
    template_name = 'user/user-about.html'

    def get_context_data(self, **kwargs):
        context = super(UserAboutPage, self).get_context_data(**kwargs)
        ratings = context['ratings'] = self.object.ratings.order_by('-contest__end_time').select_related('contest') \
            .defer('contest__description')

        context['rating_data'] = mark_safe(json.dumps([{
            'label': rating.contest.name,
            'rating': rating.rating,
            'ranking': rating.rank,
            'link': '%s#!%s' % (reverse('contest_ranking', args=(rating.contest.key,)), self.object.user.username),
            'timestamp': (rating.contest.end_time - EPOCH).total_seconds() * 1000,
            'date': date_format(timezone.localtime(rating.contest.end_time), _('M j, Y, G:i')),
            'class': rating_class(rating.rating),
            'height': '%.3fem' % rating_progress(rating.rating),
        } for rating in ratings]))

        submissions = (
            self.object.submission_set
            .annotate(date_only=TruncDate('date'))
            .values('date_only').annotate(cnt=Count('id'))
        )

        context['submission_data'] = mark_safe(json.dumps({
            date_counts['date_only'].isoformat(): date_counts['cnt'] for date_counts in submissions
        }))
        context['submission_metadata'] = mark_safe(json.dumps({
            'min_year': (
                self.object.submission_set
                .annotate(year_only=ExtractYear('date'))
                .aggregate(min_year=Min('year_only'))['min_year']
            ),
        }))
        return context


class UserProblemsPage(UserPage):
    template_name = 'user/user-problems.html'

    def get_context_data(self, **kwargs):
        context = super(UserProblemsPage, self).get_context_data(**kwargs)

        result = Submission.objects.filter(user=self.object, points__gt=0, problem__is_public=True,
                                           problem__is_organization_private=False) \
            .exclude(problem__in=self.get_completed_problems() if self.hide_solved else []) \
            .values('problem__id', 'problem__code', 'problem__name', 'problem__points', 'problem__group__full_name') \
            .distinct().annotate(points=Max('points')).order_by('problem__group__full_name', 'problem__code')

        def process_group(group, problems_iter):
            problems = list(problems_iter)
            points = sum(map(itemgetter('points'), problems))
            return {'name': group, 'problems': problems, 'points': points}

        context['best_submissions'] = [
            process_group(group, problems) for group, problems in itertools.groupby(
                remap_keys(result, {
                    'problem__code': 'code', 'problem__name': 'name', 'problem__points': 'total',
                    'problem__group__full_name': 'group',
                }), itemgetter('group'))
        ]
        breakdown, has_more = get_pp_breakdown(self.object, start=0, end=10)
        context['pp_breakdown'] = breakdown
        context['pp_has_more'] = has_more

        return context


class UserPerformancePointsAjax(UserProblemsPage):
    template_name = 'user/pp-table-body.html'

    def get_context_data(self, **kwargs):
        context = super(UserPerformancePointsAjax, self).get_context_data(**kwargs)
        try:
            start = int(self.request.GET.get('start', 0))
            end = int(self.request.GET.get('end', settings.DMOJ_PP_ENTRIES))
            if start < 0 or end < 0 or start > end:
                raise ValueError
        except ValueError:
            start, end = 0, 100
        breakdown, self.has_more = get_pp_breakdown(self.object, start=start, end=end)
        context['pp_breakdown'] = breakdown
        return context

    def get(self, request, *args, **kwargs):
        httpresp = super(UserPerformancePointsAjax, self).get(request, *args, **kwargs)
        httpresp.render()

        return JsonResponse({
            'results': utf8text(httpresp.content),
            'has_more': self.has_more,
        })


class UserDataMixin:
    @cached_property
    def data_path(self):
        return os.path.join(settings.DMOJ_USER_DATA_CACHE, '%s.zip' % self.request.profile.id)

    def dispatch(self, request, *args, **kwargs):
        if not settings.DMOJ_USER_DATA_DOWNLOAD or self.request.profile.mute:
            raise Http404()
        return super().dispatch(request, *args, **kwargs)


class UserPrepareData(LoginRequiredMixin, UserDataMixin, TitleMixin, FormView):
    template_name = 'user/prepare-data.html'
    form_class = DownloadDataForm

    @cached_property
    def _now(self):
        return timezone.now()

    @cached_property
    def can_prepare_data(self):
        return (
            self.request.profile.data_last_downloaded is None or
            self.request.profile.data_last_downloaded + settings.DMOJ_USER_DATA_DOWNLOAD_RATELIMIT < self._now or
            not os.path.exists(self.data_path)
        )

    @cached_property
    def data_cache_key(self):
        return 'celery_status_id:user_data_download_%s' % self.request.profile.id

    @cached_property
    def in_progress_url(self):
        status_id = cache.get(self.data_cache_key)
        status = task_status_by_id(status_id).status if status_id else None
        return (
            self.build_task_url(status_id)
            if status in ('PENDING', 'PROGRESS', 'STARTED')
            else None
        )

    def build_task_url(self, status_id):
        return task_status_url_by_id(
            status_id, message=_('Preparing your data...'), redirect=reverse('user_prepare_data'),
        )

    def get_title(self):
        return _('Download your data')

    def form_valid(self, form):
        self.request.profile.data_last_downloaded = self._now
        self.request.profile.save()
        status = prepare_user_data.delay(self.request.profile.id, json.dumps(form.cleaned_data))
        cache.set(self.data_cache_key, status.id)
        return HttpResponseRedirect(self.build_task_url(status.id))

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['can_prepare_data'] = self.can_prepare_data
        context['can_download_data'] = os.path.exists(self.data_path)
        context['in_progress_url'] = self.in_progress_url
        context['ratelimit'] = settings.DMOJ_USER_DATA_DOWNLOAD_RATELIMIT

        if not self.can_prepare_data:
            context['time_until_can_prepare'] = (
                settings.DMOJ_USER_DATA_DOWNLOAD_RATELIMIT - (self._now - self.request.profile.data_last_downloaded)
            )
        return context

    def post(self, request, *args, **kwargs):
        if not self.can_prepare_data or self.in_progress_url is not None:
            raise PermissionDenied()
        return super().post(request, *args, **kwargs)


class UserDownloadData(LoginRequiredMixin, UserDataMixin, View):
    def get(self, request, *args, **kwargs):
        if not os.path.exists(self.data_path):
            raise Http404()

        response = HttpResponse()

        if hasattr(settings, 'DMOJ_USER_DATA_INTERNAL'):
            url_path = '%s/%s.zip' % (settings.DMOJ_USER_DATA_INTERNAL, self.request.profile.id)
        else:
            url_path = None
        add_file_response(request, response, url_path, self.data_path)

        response['Content-Type'] = 'application/zip'
        response['Content-Disposition'] = 'attachment; filename=%s-data.zip' % self.request.user.username
        return response


@login_required
def edit_profile(request):
    if request.profile.mute:
        return generic_message(request, _("Can't edit profile"), _('Your part is silent, little toad.'), status=403)
    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=request.profile, user=request.user)
        if form.is_valid():
            with revisions.create_revision(atomic=True):
                form.save()
                revisions.set_user(request.user)
                revisions.set_comment(_('Updated on site'))

            if newsletter_id is not None:
                try:
                    subscription = Subscription.objects.get(user=request.user, newsletter_id=newsletter_id)
                except Subscription.DoesNotExist:
                    if form.cleaned_data['newsletter']:
                        Subscription(user=request.user, newsletter_id=newsletter_id, subscribed=True).save()
                else:
                    if subscription.subscribed != form.cleaned_data['newsletter']:
                        subscription.update(('unsubscribe', 'subscribe')[form.cleaned_data['newsletter']])

            perm = Permission.objects.get(codename='test_site', content_type=ContentType.objects.get_for_model(Profile))
            if form.cleaned_data['test_site']:
                request.user.user_permissions.add(perm)
            else:
                request.user.user_permissions.remove(perm)

            return HttpResponseRedirect(request.path)
    else:
        form = ProfileForm(instance=request.profile, user=request.user)
        if newsletter_id is not None:
            try:
                subscription = Subscription.objects.get(user=request.user, newsletter_id=newsletter_id)
            except Subscription.DoesNotExist:
                form.fields['newsletter'].initial = False
            else:
                form.fields['newsletter'].initial = subscription.subscribed
        form.fields['test_site'].initial = request.user.has_perm('judge.test_site')

    return render(request, 'user/edit-profile.html', {
        'require_staff_2fa': settings.DMOJ_REQUIRE_STAFF_2FA,
        'form': form, 'title': _('Edit profile'), 'profile': request.profile,
        'can_download_data': bool(settings.DMOJ_USER_DATA_DOWNLOAD),
        'has_math_config': bool(settings.MATHOID_URL),
        'ignore_user_script': True,
        'TIMEZONE_MAP': settings.TIMEZONE_MAP,
    })


@require_POST
@login_required
def generate_api_token(request):
    profile = request.profile
    with revisions.create_revision(atomic=True):
        revisions.set_user(request.user)
        revisions.set_comment(_('Generated API token for user'))
        return JsonResponse({'data': {'token': profile.generate_api_token()}})


@require_POST
@login_required
def remove_api_token(request):
    profile = request.profile
    with revisions.create_revision(atomic=True):
        profile.api_token = None
        profile.save()
        revisions.set_user(request.user)
        revisions.set_comment(_('Removed API token for user'))
    return JsonResponse({})


@require_POST
@login_required
def generate_scratch_codes(request):
    profile = request.profile
    with revisions.create_revision(atomic=True):
        revisions.set_user(request.user)
        revisions.set_comment(_('Generated scratch codes for user'))
    return JsonResponse({'data': {'codes': profile.generate_scratch_codes()}})


class UserList(QueryStringSortMixin, InfinitePaginationMixin, DiggPaginatorMixin, TitleMixin, ListView):
    model = Profile
    title = gettext_lazy('Leaderboard')
    context_object_name = 'users'
    template_name = 'user/list.html'
    paginate_by = 100
    all_sorts = frozenset(('points', 'problem_count', 'rating', 'performance_points'))
    default_desc = all_sorts
    default_sort = '-performance_points'

    def get_queryset(self):
        return (Profile.objects.filter(is_unlisted=False).order_by(self.order, 'id').select_related('user')
                .only('display_rank', 'user__username', 'username_display_override', 'points', 'rating',
                      'performance_points', 'problem_count'))

    def get_context_data(self, **kwargs):
        context = super(UserList, self).get_context_data(**kwargs)
        context['users'] = ranker(
            context['users'],
            key=attrgetter('performance_points', 'problem_count'),
            rank=self.paginate_by * (context['page_obj'].number - 1),
        )
        context['first_page_href'] = '.'
        context.update(self.get_sort_context())
        context.update(self.get_sort_paginate_context())
        return context


user_list_view = UserList.as_view()


class FixedContestRanking(ContestRanking):
    contest = None

    def get_object(self, queryset=None):
        return self.contest


def users(request):
    if request.user.is_authenticated:
        participation = request.profile.current_contest
        if participation is not None:
            contest = participation.contest
            return FixedContestRanking.as_view(contest=contest)(request, contest=contest.key)
    return user_list_view(request)


def user_ranking_redirect(request):
    try:
        username = request.GET['handle']
    except KeyError:
        raise Http404()
    user = get_object_or_404(Profile, user__username=username)
    rank = Profile.objects.filter(is_unlisted=False, performance_points__gt=user.performance_points).count()
    rank += Profile.objects.filter(
        is_unlisted=False, performance_points__exact=user.performance_points, id__lt=user.id,
    ).count()
    page = rank // UserList.paginate_by
    return HttpResponseRedirect('%s%s#!%s' % (reverse('user_list'), '?page=%d' % (page + 1) if page else '', username))


class UserLogoutView(TitleMixin, TemplateView):
    template_name = 'registration/logout.html'
    title = gettext_lazy('You have been successfully logged out.')

    def post(self, request, *args, **kwargs):
        auth_logout(request)
        return HttpResponseRedirect(request.get_full_path())


class CustomPasswordResetView(PasswordResetView):
    template_name = 'registration/password_reset.html'
    html_email_template_name = 'registration/password_reset_email.html'
    email_template_name = 'registration/password_reset_email.txt'
    extra_email_context = {'site_admin_email': settings.SITE_ADMIN_EMAIL}

    def post(self, request, *args, **kwargs):
        key = f'pwreset!{request.META["REMOTE_ADDR"]}'
        cache.add(key, 0, timeout=settings.DMOJ_PASSWORD_RESET_LIMIT_WINDOW)
        if cache.incr(key) > settings.DMOJ_PASSWORD_RESET_LIMIT_COUNT:
            return HttpResponse(_('You have sent too many password reset requests. Please try again later.'),
                                content_type='text/plain', status=429)
        return super().post(request, *args, **kwargs)


class ProfileCreationForm(ProfileForm):
    def clean_about(self):
        # Skip the solve check for new users
        return self.cleaned_data.get('about', '')


class ProfileCreationView(TitledTemplateView):
    template_name = 'registration/profile_creation.html'
    title = _('Create your profile')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if 'form' not in context:
            # Get or initialize profile
            profile = Profile.objects.filter(user=self.request.user).first()
            if profile:
                form = ProfileForm(instance=profile, user=self.request.user)
            else:
                form = ProfileCreationForm(user=self.request.user)
            context['form'] = form
        return context

    def post(self, request, *args, **kwargs):
        # Get existing profile or None
        profile = Profile.objects.filter(user=request.user).first()
        
        if profile:
            form = ProfileForm(request.POST, instance=profile, user=request.user)
        else:
            form = ProfileCreationForm(request.POST, user=request.user)
            
        if form.is_valid():
            with revisions.create_revision(atomic=True):
                profile = form.save(commit=False)
                if not hasattr(profile, 'user'):  # Only set user if this is a new profile
                    profile.user = request.user
                if not hasattr(profile, 'language'):  # Only set language if not already set
                    profile.language = Language.get_default_language()
                profile.save()
                form.save_m2m()  # Save many-to-many relationships
                revisions.set_user(request.user)
                revisions.set_comment('Updated on profile creation' if profile else 'Created on registration')
            return redirect('home')
        else:
            print("Form errors:", form.errors)  # Debug form errors
        return self.render_to_response(self.get_context_data(form=form))
