import numpy as np

def head(file, mesh):
  Node = mesh.Node
  Elmt = mesh.Elmt

  file.write('<?xml version="1.0"?>\n')
  file.write('<VTKFile type="UnstructuredGrid" version="0.1">\n')
  file.write('<UnstructuredGrid>\n')
  file.write('<Piece NumberOfPoints="'+str(np.shape(Node)[0])+'" NumberOfCells="'+str(np.shape(Elmt)[0])+'">\n')


def node(file, mesh):
  file.write('<Points>\n')
  file.write('<DataArray type="Float64" Name="nodes" NumberOfComponents="3" format="ascii">\n')
  for i in range(np.shape(mesh.Node)[0]):
    file.write(f'{mesh.Node[i,0]:.8e}  {mesh.Node[i,1]:.8e}  {mesh.Node[i,2]:.8e}'+'\n')
  file.write('</DataArray>\n')
  file.write('</Points>\n')


def elmt(file, mesh):
  file.write('<Cells>\n')
  file.write('<DataArray type="Int32" Name="connectivity" NumberOfComponents="1" format="ascii">\n')
  for i in range(np.shape(mesh.Elmt)[0]):
    file.write(f'{mesh.Elmt[i,0]:7d}{mesh.Elmt[i,1]:7d}{mesh.Elmt[i,2]:7d}{mesh.Elmt[i,3]:7d}\n')
  file.write('</DataArray>\n')
  file.write('<DataArray type="Int32" Name="offsets" NumberOfComponents="1" format="ascii">\n')
  for i in range(np.shape(mesh.Elmt)[0]):
    file.write(f'{(i+1)*4:10d}\n')
  file.write('</DataArray>\n')
  file.write('<DataArray type="Int32" Name="types" NumberOfComponents="1" format="ascii">\n')
  for i in range(np.shape(mesh.Elmt)[0]):
    file.write(f'{9:4d}\n')
  file.write('</DataArray>\n')
  file.write('</Cells>\n')


def nddata(file, mesh, dataname):
  data = mesh.nodedata

  file.write('<CellData>\n')
  file.write('<DataArray type="Float64" Name="'+dataname+'" NumberOfComponents="3" format="ascii">\n')
  for num in data:
    file.write(f'{num[0]:.8e}  {num[1]:.8e}  {num[2]:.8e}\n')
  file.write('</DataArray>\n')
  file.write('</CellData>')
  

def end(file, mesh):
  file.write('</Piece>\n')
  file.write('</UnstructuredGrid>\n')
  file.write('</VTKFile>\n')