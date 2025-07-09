import json
from myshapes import Sphere, Triangle, Plane
import numpy as np
import matplotlib.pyplot as plt

scene_fn = "scene_4.json"
res = 256

#### Scene Loader

def loadScene(scene_fn):

	with open(scene_fn) as f:
		data = json.load(f)

	spheres = []

	for sphere in data["Spheres"]:
		spheres.append(
			Sphere(sphere["Center"], sphere["Radius"], 
		 	sphere["Mdiff"], sphere["Mspec"], sphere["Mgls"], sphere["Refl"],
		 	sphere["Kd"], sphere["Ks"], sphere["Ka"]))
		
	triangles = []

	for triangle in data["Triangles"]:
		triangles.append(
			Triangle(triangle["A"], triangle["B"], triangle["C"],
			triangle["Mdiff"], triangle["Mspec"], triangle["Mgls"], triangle["Refl"],
			triangle["Kd"], triangle["Ks"], triangle["Ka"]))
	
	planes = []

	for plane in data["Planes"]:
		planes.append(
			Plane(plane["Normal"], plane["Distance"],
			plane["Mdiff"], plane["Mspec"], plane["Mgls"], plane["Refl"],
			plane["Kd"], plane["Ks"], plane["Ka"]))
	
	objects = spheres + triangles + planes

	camera = {
		"LookAt": np.array(data["Camera"]["LookAt"],),
		"LookFrom": np.array(data["Camera"]["LookFrom"]),
		"Up": np.array(data["Camera"]["Up"]),
		"FieldOfView": data["Camera"]["FieldOfView"]
	}

	light = {
		"DirectionToLight": np.array(data["Light"]["DirectionToLight"]),
		"LightColor": np.array(data["Light"]["LightColor"]),
		"AmbientLight": np.array(data["Light"]["AmbientLight"]),
		"BackgroundColor": np.array(data["Light"]["BackgroundColor"]),
	}

	return camera, light, objects

### Ray Tracer

camera, light, objects = loadScene(scene_fn)

image = np.zeros((res,res,3), dtype=np.float32)


#Gram-Schmidt
d = np.linalg.norm(camera["LookAt"]-camera["LookFrom"])

e3 = ((camera["LookAt"]-camera["LookFrom"])/d)

denominator = np.linalg.norm(np.cross(e3,camera["Up"]))

e1 = ((np.cross(e3,camera["Up"]))/denominator)

e2 = (np.cross(e1,e3))/(np.linalg.norm(np.cross(e1,e3)))

#window dimensions
d2 = np.linalg.norm(camera["LookAt"]-camera["LookFrom"])

umax = d2*np.tan(np.radians(camera["FieldOfView"]/2))
vmax = d2*np.tan(np.radians(camera["FieldOfView"]/2))

umin = -umax
vmin = -vmax

#s coordinate

du = (umax-umin)/(res+1)
dv = (vmax-vmin)/(res+1)

height = res//2
width = res//2

Ro = camera["LookFrom"]

def DetIntersection(RO, D):
	Tmin = -1
	closest = None
	for obj in objects:
		t = obj.intersect(RO, D)
		##t = -(np.dot(pn, RO) + D)/(np.dot(pn, RO))
		if Tmin == -1 and t > 0.000001:
			Tmin = t
			closest = obj
		elif 0.000001 < t < Tmin:
			Tmin = t
			closest = obj

	return Tmin, closest
#Cast Rays
for i in range(-height,height):
	for j in range(-width,width):
		s = camera["LookAt"] + du*(j+0.5)*e1 + dv*(i+0.5)*e2
		
		RayDirection = (s - Ro)/(np.linalg.norm(s - Ro))

		#Coloring stuff
		Tmin, closest = DetIntersection(Ro, RayDirection)
		
		l = (light["DirectionToLight"])/(np.linalg.norm(light["DirectionToLight"]))

		p = Ro + RayDirection*Tmin

		Tmin2, shadow = DetIntersection(p, l)

		if closest is None:
			color = light["BackgroundColor"]
			image[i+height,j+width] = np.clip(color, 0, 1)
			continue
		else:
			if shadow is not None:
				color = light["AmbientLight"]
			else:
				if isinstance(closest, Sphere):
					normal = closest.getNormal(p)
				else:
					normal = closest.getNormal()

				r = 2*(np.sum((-RayDirection)*normal))*normal - (-RayDirection)

				
		##r = r + 0.00001*RayDirection

		Tmin3, reflection = DetIntersection(p + 0.00001*r, r)

		

		if reflection is None:
			crefl = light["AmbientLight"]
		else:
			crefl = reflection.getDiffuse()

		if closest is None:
			color = light["BackgroundColor"]
		else:
			if shadow is not None:
				color = light["AmbientLight"]
			else:
				
				if isinstance(closest, Sphere):
					normal = closest.getNormal(p)
				else:
					normal = closest.getNormal()

				r = 2*(np.sum(l*normal))*(normal - l)
				cspec = (light["LightColor"]*closest.getSpecular())*(np.sum((-RayDirection)*r))**closest.getGloss()
				cdiff = (light["LightColor"]*closest.getDiffuse())*(np.sum(l*normal))
				camb = light["AmbientLight"]

				kr = closest.getRefl()

				if closest is None:
					color = light["BackgroundColor"]
				else:
					color = ((closest.getKd())*(cdiff)) + ((closest.getKs())*(cspec)) + ((closest.getKa())*(light["BackgroundColor"])) + ((kr)*(crefl))

		


		image[i+height,j+width] = np.abs(RayDirection)

		image[i+height,j+width] = np.clip(color, 0, 1)
		##image[i+height,j+width] = color





### Save and Display Output
image = np.flipud(image)
plt.imsave("output.png", image)
plt.imshow(image);plt.show()

