#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import os
sys.path.append( os.getcwd() )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import re
from importlib import import_module

latex_begin = '''
\\documentclass{standalone}
\\usepackage{sourcecodepro}
\\usepackage{forest}
\\begin{document}
\\ttfamily\\footnotesize
\\catcode`\\~=12
\\catcode`\\|=13\let|\\textbar
\\catcode`\\<=13\let<\\textless
\\catcode`\\>=13\let>\\textgreater
\\forestset{default preamble={for tree={grow'=north}}}
\\begin{forest}
'''.strip()

latex_end = '''
\\end{forest}
\\end{document}
'''.strip()

def latex_tree(tree, nodes):
  latex = "["
  if tree:
    n = tree.pop(0)
    if n in nodes:
      n = f"{n}." + nodes[n]
    latex += "{" + re.sub(r'([_$&])', r'\\\1', str(n)) + "}"
    while tree:
      latex += latex_tree(tree.pop(0), nodes)
  latex += "]"
  return latex

def latex_proof(io, proof):
  io.write( f"{latex_begin}\n" )
  io.write( "{}\n".format( latex_tree(proof['tree'], proof['nodes']) ) )
  io.write( f"{latex_end}\n" )
  return io

def load_proof(path):
  proof = import_module( re.sub(r'\.py$', '', path) )
  tree = proof.tree
  try:
    nodes = proof.nodes
  except AttributeError:
    nodes = {}
  return { 'tree': tree, 'nodes': nodes }

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i', '--input',
      required=True,
      help='Python specification of proof tree',
      )
  parser.add_argument(
      '-o', '--output',
      type=argparse.FileType('w'),
      required=True,
      help='LaTeX code for proof tree',
      )
  args = parser.parse_args()
  latex_proof(args.output, load_proof(args.input))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# vim:set et sw=2 ts=2:
