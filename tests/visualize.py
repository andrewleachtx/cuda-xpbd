# to be used as visualize.blend
import bpy
from mathutils import Quaternion, Vector
from math import radians

file_text = []

#initial_rotations = [
#    Quaternion((0, 1, 0), radians(-)),
#    Quaternion((0, 1, 0), radians(-90)),
#    Quaternion((0, 1, 0), radians(-90)),
#    Quaternion((0, 0, 1), 0),
#]

#initial_translations = [
#    Vector((0, 0, 0)),
#    Vector((0, 0, 0)),   
#    Vector((0, 0, 0)), 
#    Vector((0.35, 0.55, 0)),
#]

# uncomment to clear 
NUM_SHAPES = 10 
initial_rotations = [Quaternion() for _ in range(NUM_SHAPES)]
initial_translations = [Vector() for _ in range(NUM_SHAPES)]

with open(bpy.path.abspath("//tx.txt")) as file:
    file_text = file.readlines()

current_frame = 0
for line in file_text:
    if line.startswith("["):
        continue
    if line.startswith("#"):   
        continue
    if line.startswith("Residuals:"):
        continue
    
    if line.startswith("Step"):
        current_frame = int(line.split()[1])
    else:
        elements = line.split()
        body_id = int(elements[0]) 
        name = f"Cube.{body_id:03d}"
        
        x = float(elements[1])
        y = float(elements[2])
        z = float(elements[3])
        parsed_position = Vector((x, y, z))
        
        initial_translation = initial_translations[body_id]
        final_position = parsed_position + initial_translation  
        bpy.data.objects[name].location = final_position
        
        w = float(elements[5])
        x = float(elements[6])
        y = float(elements[7])
        z = float(elements[8])
        final_rotation = Quaternion((w, x, y, z))
        
        initial_rotation = initial_rotations[body_id]
        combined_rotation = initial_rotation @ final_rotation

        bpy.data.objects[name].rotation_mode = "QUATERNION"
        bpy.data.objects[name].rotation_quaternion = combined_rotation

        bpy.data.objects[name].keyframe_insert(data_path="location", frame=current_frame)
        bpy.data.objects[name].keyframe_insert(data_path="rotation_quaternion", frame=current_frame)

try:
    with open(bpy.path.abspath("//positions_valid_autogen.txt")) as file:
        file_text = file.readlines()
except FileNotFoundError:
    print("No validations file; skipping")
    file_text = []

current_frame = 0
for line in file_text:
    if line.startswith("Step"):
        current_frame = int(line.split()[1])
    else:
        elements = line.split()
        body_id = int(elements[0]) - 1
        name = f"Cube_valid.{body_id:03d}"
        
        x = float(elements[1])
        y = float(elements[2])
        z = float(elements[3])
        parsed_position = Vector((x, y, z))
         
        initial_translation = initial_translations[body_id]
        final_position = parsed_position + initial_translation
        bpy.data.objects[name].location = final_position
        
        w = float(elements[5])
        x = float(elements[6])
        y = float(elements[7])
        z = float(elements[8])
        final_rotation = Quaternion((w, x, y, z))
        
        initial_rotation = initial_rotations[body_id]
        combined_rotation = initial_rotation @ final_rotation

        bpy.data.objects[name].rotation_mode = "QUATERNION"
        bpy.data.objects[name].rotation_quaternion = combined_rotation

        bpy.data.objects[name].keyframe_insert(data_path="location", frame=current_frame)
        bpy.data.objects[name].keyframe_insert(data_path="rotation_quaternion", frame=current_frame)
