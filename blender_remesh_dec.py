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
  parser.add_argument('-adp', '--adaptivity', type=float, default=0.0, help="Adaptivity")
  parser.add_argument('-m', '--mode', type=str, default="BLOCKS", help="Mode")
  parser.add_argument('-oct', '--octree_depth', type=int, default=4, help="Mode")
  parser.add_argument('-s', '--scale', type=float, default=0.9, help="Scale")
  parser.add_argument('-sh', '--sharpness', type=float, default=1.0, help="Threshold")
  parser.add_argument('-t', '--threshold', type=float, default=1.0, help="Threshold")
  parser.add_argument('-smooth', '--use_smooth_shade', type=bool,  default=False, help="Smooth or Flat")
  parser.add_argument('-v', '--voxel_size', type=float, default=0.1, help="Voxel Size")
  parser.add_argument('-dec', '--decimate_ratio', type=float, default=1.0 help="Ratio of reduction, Example: 0.5 mean half number of faces ")

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

  # Decimate
  modifier = obj.modifiers.new(modifierName,'DECIMATE')
  modifier.ratio = 1.0
  modifier.use_collapse_triangulate = True
  bpy.ops.object.modifier_apply(modifier=modifierName)
  print("{} has {} verts, {} edges, {} polys after decimation".format(obj.name, len(obj.data.vertices), len(obj.data.edges), len(obj.data.polygons)))

  # Remesh
  modifier = obj.modifiers.new(modifierName,'REMESH')
  modifier.mode = "SMOOTH"
  modifier.octree_depth = 6
  bpy.ops.object.modifier_apply(modifier=modifierName)

  print("{} has {} verts, {} edges, {} polys after remesh".format(obj.name, len(obj.data.vertices), len(obj.data.edges), len(obj.data.polygons)))

  # Back to Tris
  bpy.ops.object.editmode_toggle()
  bpy.ops.mesh.select_all(action='SELECT')
  bpy.ops.mesh.quads_convert_to_tris()
  bpy.ops.mesh.normals_make_consistent()
  bpy.ops.object.editmode_toggle()

bpy.ops.export_scene.obj(filepath=output_model, path_mode="STRIP")
print('\n Process of CleanUp Finished ...')

# blender -b -P blender_cleanup.py -- -in mesh.obj -out clean.obj
