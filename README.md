# ncc-erp-contest

## DMOJ SITE

- Django MVC: Contest, Problem, User,...
- Path: `site`
- Python 3.8

### Basic Check

- Go to site folder.
- Create file 'local_settings.py' inside `site\dmoj` and modify it with your settings. Please see the example.local_settings.py
- Please run ./make_style.sh to gen styles file
- Run `python3 manager.py collectstatic`, `python3 manager.py compilejsi18n`, `python3 manager.py compilemessages` to generate static files and i18n translated
- `site\dmoj\settings.py`: All setting of Site, check and change
  - STATIC_ROOT
  - DATABASES
  - GOOGLE AUTH Configuration
  - Email Configuration
- `site\dmoj\my.cnf`: Information database

## Judge Server

- To run submission code
- Config languages run
- Stored Problems
