#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Mat Leonard'
SITENAME = 'Matatat.org'
SITEURL = ''

PATH = 'content'
STATIC_PATHS = ['images']

THEME = '/Users/mat/.pelican-themes/octopress'
MENUITEMS = [('Archives', '/archives.html'),]
NEWEST_FIRST_ARCHIVES = False

DEFAULT_DATE_FORMAT = '%b %d, %Y'
TIMEZONE = 'US/Pacific'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
# FEED_ALL_ATOM = None
# CATEGORY_FEED_ATOM = None
# TRANSLATION_FEED_ATOM = None
# AUTHOR_FEED_ATOM = None
# AUTHOR_FEED_RSS = None

# Blogroll
# LINKS = (('Pelican', 'http://getpelican.com/'),
#          ('Python.org', 'http://python.org/'),
#          ('Jinja2', 'http://jinja.pocoo.org/'),)

# Social widget
#SOCIAL = (('Twitter', 'https://twitter.com/MCLeopard'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

PLUGIN_PATHS = ['pelican-plugins']
PLUGINS = ['summary', 'render_math', 'liquid_tags.video']