"""
Warnings d'un run pycte -> warning.log.

Tout warning passe par warn(message) : affiche a la console et ecrit dans warning.log, a chaque fois qu'il est emis.
Les warnings Python (warnings.warn de pycte, numpy, pandas, ...) sont rediriges vers warn() pendant le run, tous et
a chaque occurrence (filtre "always" : ni les repetitions ni les DeprecationWarning ne sont masquees).

Les messages restent en memoire et sont ecrits en un seul bloc : start() cree warning.log (en-tete du run + warnings
deja emis), flush() est appele par engine.py a la fin de chaque pas de couplage, end() en fin de run, y compris sur
sys.exit ou erreur.

Workers (ProcessPoolExecutor) : submit(executor, fn, ...) au lieu de executor.submit(fn, ...) renvoie au processus
principal les warnings emis dans le worker ; le Future rendu s'utilise comme celui de executor.submit.
"""

import os
import threading
import warnings
from concurrent.futures import Future

LOG_NAME = "warning.log"

_lock = threading.Lock()
_pending = []           # messages pas encore ecrits dans warning.log
_started = False        # warning.log cree pour le run courant
_console = True         # affichage console (coupe dans les workers : le processus principal affiche)
_saved = None           # filtres et affichage des warnings Python avant begin(), rendus par end()


def warn(message):
    """Ajoute `message` aux warnings a ecrire dans warning.log et l'affiche."""
    message = str(message).rstrip()
    with _lock:
        _pending.append(message)
    if _console:
        print(f"\nWARNING {message}", flush=True)


def _fromPython(message, category, filename, lineno, file=None, line=None):
    warn(f"{category.__name__} ({os.path.basename(filename)}:{lineno}) : {message}")


def begin(console=True):
    """Debut de run : oublie les warnings d'un run precedent et capte tous les warnings Python."""
    global _started, _console, _saved
    with _lock:
        _pending.clear()
        _started = False
    _console = console
    if _saved is None:
        _saved = warnings.catch_warnings()
        _saved.__enter__()
    warnings.showwarning = _fromPython
    warnings.simplefilter("always")


def _write(mode, header, centralDict):
    with _lock:
        messages = _pending[:]
        _pending.clear()
    if mode == "a" and not messages:
        return
    with open(LOG_NAME, mode) as log:
        log.write(header + "".join(message.replace("\n", "\n    ") + "\n" for message in messages))
    if centralDict is not None:
        centralDict['warningNbr'] = centralDict.get('warningNbr', 0) + len(messages)


def start(header, centralDict=None):
    """Cree warning.log : `header` puis les warnings emis jusque-la, comptes dans centralDict['warningNbr']."""
    global _started
    _started = True
    _write("w", header, centralDict)


def flush(centralDict=None):
    """Ecrit en un seul bloc les warnings en attente (rien avant start()), comptes dans centralDict['warningNbr']."""
    if _started:
        _write("a", "", centralDict)


def end():
    """Fin de run, normale ou non : ecrit les warnings restants et rend l'affichage et les filtres des warnings Python."""
    global _saved
    flush()
    if _saved is not None:
        _saved.__exit__(None, None, None)
        _saved = None


def _inWorker(fn, args, kwargs):
    """Execute fn dans un worker ; renvoie (resultat, warnings emis pendant l'appel)."""
    begin(console=False)
    try:
        result = fn(*args, **kwargs)
    finally:
        end()
    return result, list(_pending)


def submit(executor, fn, *args, **kwargs):
    """executor.submit(fn, *args, **kwargs) ; les warnings du worker reviennent dans warning.log."""
    inner = executor.submit(_inWorker, fn, args, kwargs)
    outer = Future()

    def relay(done):
        try:
            result, messages = done.result()
        except BaseException as error:
            outer.set_exception(error)
            return
        for message in messages:
            warn(message)
        outer.set_result(result)

    inner.add_done_callback(relay)
    return outer
