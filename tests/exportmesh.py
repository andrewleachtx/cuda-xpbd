# to be used in Blender to export all mesh instances in the scene
import bpy
import os

# https://blender.stackexchange.com/a/309888/221840
export_dir = "C:/Users/andre/Desktop/code/graphics/randblender/output"

if not os.path.exists(export_dir):
    os.makedirs(export_dir)

for obj in bpy.context.scene.objects:
    if obj.type == "MESH":

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        
        # export_path = os.path.join(export_dir, f"{obj.name}.obj")
        export_path = os.path.join(export_dir, f"bowl{obj.name.split('.')[1]}.obj")
        
        bpy.ops.wm.obj_export(
            filepath=export_path,
            export_selected_objects=True,
            forward_axis='NEGATIVE_Z',
            up_axis='Y',
            apply_modifiers=True,
            export_uv=True,
            export_normals=True,
            export_materials=True
        )

print(f"Exported all objects to {export_dir}")