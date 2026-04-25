# justfile for cartographer3d-plugin
set dotenv-load

printer_host := env("PRINTER_HOST", "pi@printer.local")
project_name := file_name(justfile_directory())

install-copy:
    rsync -azqP --delete --filter=':- .gitignore' --filter='- .jj' './' '{{printer_host}}:~/{{project_name}}/'

install-pip: install-copy
    ssh {{printer_host}} '$HOME/klippy-env/bin/pip install --upgrade --disable-pip-version-check -qU $HOME/{{project_name}}'

install: install-pip
    ssh {{printer_host}} '$HOME/klippy-env/bin/pip list --disable-pip-version-check | grep cartographer | cut -d " " -f 2'

restart: install
    ssh {{printer_host}} 'sudo systemctl stop klipper && rm ~/printer_data/comms/* ~/printer_data/logs/klippy.log && sudo systemctl start klipper'
