from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.align import Align
from rich.tree import Tree
import subprocess
import shutil
import os
import sys
import venv

console = Console()

def check_pnpm():
    if shutil.which("pnpm") is None:
        console.print("[bold red]Error:[/bold red] 'pnpm' no está instalado. Por favor instálalo antes de continuar.")
        input("Presiona ENTER para volver al menú.")
        return False
    return True

def ensure_virtualenv():
    venv_path = ".venv"
    if not os.path.isdir(venv_path):
        console.print("[yellow]Entorno virtual no encontrado. Creando entorno virtual...[/yellow]")
        with console.status("Creando entorno virtual...", spinner="dots"):
            venv.create(venv_path, with_pip=True)

    requirements = os.path.join("main", "requirements.txt")
    if os.path.isfile(requirements):
        console.print("[yellow]Instalando dependencias desde requirements.txt...[/yellow]")
        pip_executable = os.path.join(venv_path, "bin", "pip") if os.name != "nt" else os.path.join(venv_path, "Scripts", "pip.exe")
        with console.status("Instalando dependencias...", spinner="dots"):
            subprocess.run([pip_executable, "install", "-r", requirements], check=True)
    
    rust_path = os.path.join("main", "rust")

    if os.path.isdir(rust_path):
        console.print("[yellow]Compilando el módulo Rust...[/yellow]")
        with console.status("Compilando módulo Rust...", spinner="dots"):
            subprocess.run(["maturin", "develop", "--release"], cwd=rust_path, check=True)


def dev():
    if not check_pnpm():
        return

    while True:
        console.clear()
        console.print(Panel("[bold green]Modo desarrollo[/bold green]", expand=False))

        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Opción", style="bold green", justify="center")
        table.add_column("Tarea", style="white")

        table.add_row("1", "Preparar entorno virtual + instalar dependencias")
        table.add_row("2", "Levantar solo API (FastAPI)")
        table.add_row("3", "Levantar solo interfaz web (UI)")
        table.add_row("4", "Levantar aplicación completa (API + UI)")
        table.add_row("0", "Volver al menú principal")

        console.print(Align.center(table))
        choice = Prompt.ask("\n[bold yellow]Selecciona una opción[/bold yellow]", choices=["1", "2", "3", "4", "0"])

        if choice == "1":
            try:
                ensure_virtualenv()
                console.print("[bold green]✅ Entorno preparado correctamente.[/bold green]")
            except subprocess.CalledProcessError:
                console.print("[bold red]❌ Error al preparar entorno virtual.[/bold red]")
            input("\nPresiona ENTER para continuar.")

        elif choice == "2":
            try:
                console.print("[cyan]🚀 Levantando FastAPI con Uvicorn en [bold]http://127.0.0.1:8000[/bold]...[/cyan]")
                subprocess.run(["uvicorn", "src.api:app", "--reload"], cwd="main", check=True)
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Proceso interrumpido por el usuario.[/bold yellow]")
            except subprocess.CalledProcessError:
                console.print("[bold red]❌ Error al ejecutar uvicorn[/bold red]")
            input("\nPresiona ENTER para continuar.")

        elif choice == "3":
            web_dir = "web"
            if not os.path.isfile(os.path.join(web_dir, "package.json")):
                console.print("[bold red]❌ No se encontró package.json en 'web/'.[/bold red]")
                input("Presiona ENTER para continuar.")
                continue

            try:
                with console.status("[cyan]Instalando dependencias con pnpm...[/cyan]", spinner="dots"):
                    subprocess.run(["pnpm", "install"], cwd=web_dir, check=True)
            except subprocess.CalledProcessError:
                console.print("[bold red]❌ Error en pnpm install[/bold red]")
                input("Presiona ENTER para continuar.")
                continue

            try:
                console.print("[cyan]🚀 Ejecutando pnpm run dev en [bold]http://localhost:5173[/bold]...[/cyan]")
                subprocess.run(["pnpm", "run", "dev"], cwd=web_dir, check=True)
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Proceso interrumpido por el usuario.[/bold yellow]")
            except subprocess.CalledProcessError:
                console.print("[bold red]❌ Error al ejecutar pnpm run dev[/bold red]")
            input("\nPresiona ENTER para continuar.")

        elif choice == "4":
            try:
                ensure_virtualenv()
            except subprocess.CalledProcessError:
                console.print("[bold red]❌ Error preparando entorno virtual[/bold red]")
                input("Presiona ENTER para continuar.")
                continue

            web_dir = "web"
            if not os.path.isfile(os.path.join(web_dir, "package.json")):
                console.print("[bold red]❌ No se encontró package.json en 'web/'.[/bold red]")
                input("Presiona ENTER para continuar.")
                continue

            try:
                subprocess.run(["pnpm", "install"], cwd=web_dir, check=True)
            except subprocess.CalledProcessError:
                console.print("[bold red]❌ Error en pnpm install[/bold red]")
                input("Presiona ENTER para continuar.")
                continue

            try:
                console.print("[green]🚀 Levantando API en http://127.0.0.1:8000[/green]")
                api_proc = subprocess.Popen(["uvicorn", "src.api:app", "--reload"], cwd="main")

                console.print("[green]🚀 Levantando UI en http://localhost:5173[/green]")
                ui_proc = subprocess.Popen(["pnpm", "run", "dev"], cwd=web_dir)

                console.print("\n[bold green]✅ Aplicación en modo desarrollo ejecutándose.[/bold green]")
                console.print("[cyan]Presiona CTRL+C en esta terminal para detener los procesos.[/cyan]")

                api_proc.wait()
                ui_proc.wait()

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Deteniendo procesos...[/bold yellow]")
                api_proc.terminate()
                ui_proc.terminate()
                api_proc.wait()
                ui_proc.wait()
            break

        elif choice == "0":
            break




def prod():
    console.print("\n[bold blue]Modo producción activado[/bold blue]")
    try:
        with console.status("[bold]Construyendo contenedores...[/bold]", spinner="dots"):
            subprocess.run(["docker-compose", "build"], check=True)

        console.print("[bold]Levantando servicios...[/bold]")
        subprocess.run(["docker-compose", "up"], check=True)
    except subprocess.CalledProcessError:
        console.print("[bold red]Error en Docker Compose[/bold red]")

    input("\nPresiona ENTER para volver al menú.")

def tree():
    if not os.path.isdir("main"):
        console.print("[bold red]El directorio 'main/' no existe.[/bold red]")
        input("\nPresiona ENTER para volver al menú.")
        return
      
    exclude_dirs = {"__pycache__", ".venv", "venv", ".git", "target", "node_modules", ".idea", ".vscode"}

    file_icons = {
        ".py": "🐍",
        ".csv": "📈",
        ".txt": "📄",
        "Dockerfile": "🐳",

        ".rs": "🦀",
        ".toml": "📦",
        ".lock": "🔒",

    }

    root_tree = Tree("🌲 [bold blue]main/[/bold blue]")

    def add_nodes(tree_node, current_path):
        try:
            entries = sorted(os.listdir(current_path))
        except PermissionError:
            return

        for entry in entries:
            if entry in exclude_dirs or entry.startswith("."):
                continue
            full_path = os.path.join(current_path, entry)
            if os.path.isdir(full_path):
                sub_tree = tree_node.add(f"📁 [bold]{entry}[/bold]")
                add_nodes(sub_tree, full_path)
            else:
                ext = os.path.splitext(entry)[1]
                icon = file_icons.get(ext, "📄")
                if entry == "Dockerfile":
                    icon = file_icons["Dockerfile"]
                tree_node.add(f"{icon} {entry}")

    add_nodes(root_tree, "main")
    console.print(root_tree)
    input("\nPresiona ENTER para volver al menú.")



def menu():
    console.clear()
    title = Panel("[bold cyan]Asistente de instalación y ejecución del proyecto[/bold cyan]", expand=False)
    console.print(Align.center(title))

    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("Opción", style="bold green", justify="center")
    table.add_column("Descripción", style="white")

    table.add_row("1", "Modo desarrollo")
    table.add_row("2", "Modo producción")
    table.add_row("3", "Ver árbol del directorio [bold]main/[/bold]")
    table.add_row("0", "Salir")

    console.print(Align.center(table))
    return Prompt.ask("\n[bold yellow]Selecciona una opción[/bold yellow]", choices=["1", "2", "3", "0"])

def main():
    while True:
        option = menu()
        if option == "1":
            dev()
        elif option == "2":
            prod()
        elif option == "3":
            tree()
        elif option == "0":
            console.print("\n[bold red]Saliendo del asistente...[/bold red]")
            sys.exit()

if __name__ == "__main__":
    main()
