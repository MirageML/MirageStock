import bpy
checked = set()
selected_objects = bpy.context.selected_objects
for selected_object in selected_objects:
    if selected_object.type != 'MESH':
        continue


#!/usr/bin/python3
import bpy
import sys
import time
import argparse



def get_args():
  parser = argparse.ArgumentParser()

  # get all script args
  _, all_arguments = parser.parse_known_args()
  double_dash_index = all_arguments.index('--')
  script_args = all_arguments[double_dash_index + 1: ]

  # add parser rules
  parser.add_argument('-in', '--inm', help="Original Model")
  parser.add_argument('-out', '--outm', help="Decimated output file")
  parser.add_argument('-quads', '--convert_quads', default=True, help="Convert Quads")
  parsed_script_args, _ = parser.parse_known_args(script_args)
  return parsed_script_args

args = get_args()
input_model = str(args.inm)
print(input_model)

output_model = str(args.outm)
print(output_model)

print('\n Clearing blender scene (default garbage...)')
# deselect all
bpy.ops.object.select_all(action='DESELECT')

# selection
# bpy.data.objects['Camera'].select = True
bpy.ops.object.delete({"selected_objects":[o for o in bpy.context.scene.objects]} )

# remove it
# bpy.ops.object.delete()

# Clear Blender scene
# select objects by type
for o in bpy.data.objects:
    if o.type == 'MESH':
        o.select = True
    else:
        o.select = False

# call the operator once
bpy.ops.object.delete()

print('\n Beginning the process of CleanUp using Blender Python API ...')
bpy.ops.import_scene.obj(filepath=input_model)
print('\n Obj file imported successfully ...')
modifierName='DecimateMod'

print('\n Creating and object list and adding meshes to it ...')
objectList=bpy.data.objects
meshes = []
for obj in objectList:
  if(obj.type == "MESH"):
    meshes.append(obj)

print("{} meshes".format(len(meshes)))

checked = set()

for i, obj in enumerate(meshes):
  bpy.context.view_layer.objects.active = obj
  print("{}/{} meshes, name: {}".format(i, len(meshes), obj.name))
  print("{} has {} verts, {} edges, {} polys".format(obj.name, len(obj.data.vertices), len(obj.data.edges), len(obj.data.polygons)))
  meshdata = obj.data
  if meshdata in checked:
    continue
  else:
    checked.add(meshdata)

  bpy.ops.object.editmode_toggle()
  bpy.ops.mesh.select_all(action='SELECT')

  # Clean Up Stuff
  bpy.ops.mesh.remove_doubles()
  bpy.ops.mesh.fill_holes() # Fill holes
  bpy.ops.mesh.face_make_planar() # Flatten faces
  bpy.ops.mesh.vert_connect_nonplanar() # Split non-planar faces
  # bpy.ops.mesh.vert_connect_concave() # Make all concave faces convex
  bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=True) # Delete loose vertices, faces and edges
  bpy.ops.mesh.dissolve_degenerate() # Dissolve degenerate faces zero area
  bpy.ops.mesh.dissolve_limited() # Dissolve faces with an angle less than 30 degrees

  # Better
  if args.convert_quads:
    bpy.ops.mesh.tris_convert_to_quads()
  bpy.ops.mesh.normals_make_consistent()

  bpy.ops.object.editmode_toggle()

  print("{} has {} verts, {} edges, {} polys after cleanup".format(obj.name, len(obj.data.vertices), len(obj.data.edges), len(obj.data.polygons)))

bpy.ops.export_scene.obj(filepath=output_model, path_mode="STRIP")
print('\n Process of CleanUp Finished ...')

# blender -b -P blender_cleanup.py -- -in mesh.obj -out clean.obj
