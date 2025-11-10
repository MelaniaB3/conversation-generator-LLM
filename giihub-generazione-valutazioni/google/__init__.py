# File: google/__init__.py

# Questo file rende la cartella 'google' un pacchetto Python.
# La riga seguente espone il nostro modulo 'generativeai' locale,
# permettendo di importarlo con 'import google.generativeai'.

from . import generativeai

__all__ = ["generativeai"]