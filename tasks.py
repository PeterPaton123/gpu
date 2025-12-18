from invoke import Context, task

@task
def poetry_env_setup(c: Context) -> None:
	c.run("echo 'Setting up poetry environment...'")
	c.run("poetry self update")
	c.run("poetry env use /bin/python3.12")
	c.run("poetry config virtualenvs.in-project true --local")

@task
def start_env_prod(c: Context) -> None:
	# Runs with $inv start-env-prod
	poetry_env_setup(c)
	c.run("poetry install")
