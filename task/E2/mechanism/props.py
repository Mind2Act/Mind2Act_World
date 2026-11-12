"""Selected E2 task geometry, with shared station/robot and resolved font asset.

Derived from training source/scene.py SHA256
67014ae373605e675dc27b49429f2332a1518bb3bf6854caece266872fb0bd2f.
No robot controller, station authoring, camera attachment or private source path.
"""
import json
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdShade, UsdPhysics, PhysxSchema
from .geometry import rounded, cylinder, material
from .objects import make_object

TABLE_Z=.768
CAP_Z=TABLE_Z+.034
PRESS_Z=CAP_Z+.008
PRESS_OFFSET_Y=-.020

def label_material(stage,path,file):
    mat=UsdShade.Material.Define(stage,path)
    shader=UsdShade.Shader.Define(stage,path+'/Shader');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.7)
    uv=UsdShade.Shader.Define(stage,path+'/UV');uv.CreateIdAttr('UsdPrimvarReader_float2')
    uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    tx=UsdShade.Shader.Define(stage,path+'/Texture');tx.CreateIdAttr('UsdUVTexture')
    tx.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(file)))
    tx.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    tx.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
    tx.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
    for key in ['diffuseColor','emissiveColor']:
        shader.CreateInput(key,Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(),'rgb')
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    return mat,tx


def digit_texture(output, digit, font_path):
    path = output / f'digit_{digit}.png'
    image = Image.new('RGB', (512, 320), (236, 237, 222))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 260)
    while draw.textbbox((0,0),str(digit),font=font)[2]>460:
        font=ImageFont.truetype(str(font_path),font.size-4)
    draw.text((256, 155), str(digit), font=font, fill=(23, 46, 54), anchor='mm')
    image.save(path)
    return path


def plaque(stage, path, center, size, material):
    x, y, z = center
    w, h = size
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(x-w/2,y-h/2,z),(x+w/2,y-h/2,z),(x+w/2,y+h/2,z),(x-w/2,y+h/2,z)])
    mesh.CreateFaceVertexCountsAttr([4]); mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
    mesh.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set([(0,0),(1,0),(1,1),(0,1)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def build_props(stage, output, case, font_path):
    """Author only the selected training task props into the shared station."""
    output.mkdir(parents=True,exist_ok=True)
    mats={
      'aluminium':material(stage, 'EKAluminium',(.44,.49,.52),.68,.43),
      'black':material(stage, 'EKBlack',(.022,.030,.035),.05,.65),
      'white':material(stage, 'EKWhite',(.85,.86,.82),.04,.58),
      'wood':material(stage, 'EKWood',(.46,.23,.09),.0,.63),
      'steel':material(stage, 'EKSteel',(.18,.25,.29),.72,.32),
      'ivory':material(stage, 'EKIvory',(.88,.81,.65),.03,.65),
      'gold':material(stage, 'EKDuckYellow',(.90,.57,.12),.08,.49),
      'orange':material(stage, 'EKOrange',(.75,.25,.055),.03,.61),
      'brass':material(stage, 'EKBrass',(.61,.43,.16),.75,.34),
    }
    rounded(stage,'/World/EKeyboard/Deck',(0,-.065,TABLE_Z+.006),(.340,.290,.012),.018,mats['aluminium'])
    rounded(stage,'/World/EKeyboard/DeckInset',(0,-.065,TABLE_Z+.013),(.328,.278,.006),.014,mats['black'])
    buttons=[];registration_labels=[]
    for i,(kind,xy) in enumerate(zip(case['objects'],case['slot_xy_m'])):
        root=f'/World/EKeyboard/Key{i}';UsdGeom.Xform.Define(stage,root)
        base=UsdGeom.Xform.Define(stage,root+'/Base');base.AddTranslateOp().Set(Gf.Vec3d(*xy,TABLE_Z))
        UsdPhysics.RigidBodyAPI.Apply(base.GetPrim()).CreateKinematicEnabledAttr(True)
        rounded(stage,root+'/Base/Frame',(0,0,.016),(.094,.087,.022),.010,mats['aluminium'])
        for ix,x in enumerate([-.034,.034]):
            for iy,y in enumerate([-.030,.030]):cylinder(stage,root+f'/Base/Spring{ix}{iy}',(x,y,.027),.003,.022,mats['steel'])
        cap=UsdGeom.Xform.Define(stage,root+'/Cap');cap.AddTranslateOp().Set(Gf.Vec3d(*xy,CAP_Z))
        UsdPhysics.RigidBodyAPI.Apply(cap.GetPrim()).CreateRigidBodyEnabledAttr(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(cap.GetPrim()).CreateDisableGravityAttr(True)
        PhysxSchema.PhysxContactReportAPI.Apply(cap.GetPrim()).CreateThresholdAttr(0.)
        UsdPhysics.MassAPI.Apply(cap.GetPrim()).CreateMassAttr(.12)
        rounded(stage,root+'/Cap/Top',(0,0,0),(.085,.078,.012),.010,mats['white'])
        col=UsdGeom.Cube.Define(stage,root+'/Cap/Collision');col.CreateSizeAttr(1)
        col.AddScaleOp().Set(Gf.Vec3f(.085,.078,.012))
        UsdPhysics.CollisionAPI.Apply(col.GetPrim()).CreateCollisionEnabledAttr(True)
        PhysxSchema.PhysxCollisionAPI.Apply(col.GetPrim()).CreateContactOffsetAttr(.0001)
        UsdGeom.Imageable(col.GetPrim()).MakeInvisible()
        land=cylinder(stage,root+'/Cap/PressLand',(0,PRESS_OFFSET_Y,.007),.014,.002,mats['black'])
        UsdPhysics.CollisionAPI.Apply(land.GetPrim()).CreateCollisionEnabledAttr(True)
        PhysxSchema.PhysxCollisionAPI.Apply(land.GetPrim()).CreateContactOffsetAttr(.0001)
        lm, _ = label_material(stage, root+'/NumberMaterial', digit_texture(output, i+1, font_path))
        number = plaque(stage, root+'/Cap/Number', (0, 0, .0065), (.056, .051), lm)
        UsdGeom.Imageable(land.GetPrim()).MakeInvisible()
        registration_labels.append(number)
        joint=UsdPhysics.PrismaticJoint.Define(stage,root+'/SpringJoint')
        joint.CreateBody0Rel().SetTargets([root+'/Base']);joint.CreateBody1Rel().SetTargets([root+'/Cap'])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0,0,.034));joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
        joint.CreateAxisAttr('Z');joint.CreateLowerLimitAttr(-.008);joint.CreateUpperLimitAttr(0.)
        drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear')
        drive.CreateStiffnessAttr(650.);drive.CreateDampingAttr(5.);drive.CreateTargetPositionAttr(0.);drive.CreateMaxForceAttr(30.)
        buttons.append(root)
    # A single plain hinged cover conceals the association-to-sequence reset.
    rounded(stage, '/World/EKeyboard/Tray', (0,.215,TABLE_Z+.006), (.780,.230,.012), .010, mats['white'])
    lid = UsdGeom.Xform.Define(stage, '/World/EKeyboard/Cover')
    lid.AddTranslateOp().Set(Gf.Vec3d(0,.330,.855))
    lid_rotation = lid.AddRotateXOp(); lid_rotation.Set(-105.)
    rounded(stage, str(lid.GetPath())+'/Panel', (0,-.1375,0), (.790,.275,.006), .006, mats['white'])
    registration = UsdGeom.Xform.Define(stage, '/World/EKeyboard/Registration')
    sequence = UsdGeom.Xform.Define(stage, '/World/EKeyboard/Sequence')
    for index, (kind, digit, xy) in enumerate(zip(case['entities'], case['entity_digits'], case['registration_xy_m'])):
        obj = make_object(stage, str(registration.GetPath())+f'/Object{index}', kind, mats)
        obj.AddTranslateOp().Set(Gf.Vec3d(*xy, TABLE_Z+.013))
        obj.AddScaleOp().Set(Gf.Vec3f(case['object_scale']))
        card_material, _ = label_material(stage, str(registration.GetPath())+f'/CardMaterial{index}', digit_texture(output, digit, font_path))
        rounded(stage, str(registration.GetPath())+f'/Card{index}', (xy[0],.155,TABLE_Z+.014), (.070,.047,.003), .002, mats['white'])
        plaque(stage, str(registration.GetPath())+f'/Digit{index}', (xy[0],.155,TABLE_Z+.016), (.066,.043), card_material)
    for index,(token,xy) in enumerate(zip(case['expression_layout_tokens'],case['expression_layout_xy_m'])):
        if type(token) is int:
            obj=make_object(stage,str(sequence.GetPath())+f'/Operand{index}',case['entities'][token],mats)
            obj.AddTranslateOp().Set(Gf.Vec3d(*xy,TABLE_Z+.013))
            obj.AddScaleOp().Set(Gf.Vec3f(case['object_scale']))
        else:
            filename=output/f'symbol_{index}.png'
            image=Image.new('RGB',(512,640),(236,237,222))
            draw=ImageDraw.Draw(image)
            font=ImageFont.truetype(str(font_path),470)
            draw.text((256,310),token,font=font,fill=(23,46,54),anchor='mm')
            image.save(filename)
            symbol_material,_=label_material(stage,str(sequence.GetPath())+f'/SymbolMaterial{index}',filename)
            rounded(stage,str(sequence.GetPath())+f'/SymbolCard{index}',(xy[0],xy[1],TABLE_Z+.014),(case['symbol_width_m'],.067,.002),.001,mats['white'])
            plaque(stage,str(sequence.GetPath())+f'/SymbolPrint{index}',(xy[0],xy[1],TABLE_Z+.0152),(case['symbol_width_m']-.001,.065),symbol_material)
    UsdGeom.Imageable(sequence.GetPrim()).MakeInvisible()
    # Auxiliary *RGB* sensor gives stable object crops; no identity labels.
    top=UsdGeom.Camera.Define(stage,'/World/EKeyboard/ObjectCamera')
    top.AddTranslateOp().Set(Gf.Vec3d(0,.095,1.50))
    top.CreateProjectionAttr('orthographic');top.CreateHorizontalApertureAttr(10.);top.CreateVerticalApertureAttr(7.5)
    top.CreateClippingRangeAttr(Gf.Vec2f(.01,10));
    from mind2act_eval.sim.asset_relocation import export_relative_layer
    export_relative_layer(stage.GetRootLayer(), output/'scene.usda')
    (output/'case.private.json').write_text(json.dumps(case,indent=2))
    return dict(buttons=buttons, registration=registration, sequence=sequence, lid_rotation=lid_rotation)
