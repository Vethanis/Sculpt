# Sculpt

A port of my meshing project from C++ to the Unity job system. This program generates triangles approximating the surface of the distance function the user inputs using brushes. Many distance functions may be input, therefore the distance functions are referenced in the leaves of an octree. Once a leaf is changed, a compute job is started to update the leaf mesh.


## **Video**

[![](https://i.imgur.com/MevYAJT.png)](https://vimeo.com/300385552)

