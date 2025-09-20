package main

import (
	"math"

	rl "github.com/gen2brain/raylib-go/raylib"
)

const WIDTH int32 = 700
const HEIGHT int32 = 700
const EPS = 0.001

type Vec3 struct {
	X, Y, Z float64
	Magn    *float64
}

func NewVector(x, y, z float64) Vec3 {
	v := Vec3{x, y, z, nil}
	magn := v.Dot(v)
	v.Magn = &magn
	return v
}

func assert(cond bool, msg string) {
	if !cond {
		panic(msg)
	}
}

func (v Vec3) Add(that Vec3) Vec3 {
	return NewVector(that.X+v.X, that.Y+v.Y, that.Z+v.Z)
}

func (v Vec3) Sub(that Vec3) Vec3 {
	return NewVector(v.X-that.X, v.Y-that.Y, v.Z-that.Z)
}

func (v Vec3) Cross(that Vec3) Vec3 {
	X := v.Y*that.Z - v.Z*that.Y
	Y := v.Z*that.X - v.X*that.Z
	Z := v.X*that.Y - v.Y*that.X

	return NewVector(X, Y, Z)
}

func (v Vec3) Dot(that Vec3) float64 {
	x := that.X * v.X
	y := that.Y * v.Y
	z := that.Z * v.Z
	return x + y + z
}

func (v Vec3) Len() float64 {
	r := math.Sqrt(v.Dot(v))
	return r
}

func (v Vec3) Norm() Vec3 {
	n := v.Len()
	if n == 0 {
		return v
	}

	return v.Div(n)
}

func (v Vec3) Mul(n float64) Vec3 {
	return NewVector(v.X*n, v.Y*n, v.Z*n)
}

func (v Vec3) Div(n float64) Vec3 {
	return NewVector(v.X/n, v.Y/n, v.Z/n)
}

func (v Vec3) toArray() [3]float64 {
	return [3]float64{v.X, v.Y, v.Z}
}

type Color struct {
	R, G, B float64
}

func NewColor(r, g, b uint8) Color {
	return Color{float64(r), float64(g), float64(b)}
}

func (c Color) Add(that Color) Color {
	return Color{c.R + that.R, c.G + that.G, c.B + that.B}
}

func norm(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 255 {
		return 255
	}
	return math.Round(x)
}

func (c Color) toRlColor() rl.Color {
	r := norm(c.R)
	g := norm(c.G)
	b := norm(c.B)
	return rl.NewColor(uint8(r), uint8(g), uint8(b), 255)
}

func (c Color) Scale(val float64) Color {
	return Color{c.R * val, c.G * val, c.B * val}
}

type ObjectType int

const (
	SphereType ObjectType = iota
	TriangleType
)

type Object interface {
	GetColor() Color
	GetSpecular() float64
	GetClarity() float64
	GetType() ObjectType
}

func isSphere(obj Object) bool {
	return obj.GetType() == SphereType
}

func isTriangle(obj Object) bool {
	return obj.GetType() == TriangleType
}

type Sphere struct {
	Center    Vec3
	Radius    float64
	RadiusSqr float64

	specular float64
	color    Color
	clarity  float64
}

func (s Sphere) GetColor() Color {
	return s.color
}

func (s Sphere) GetNorm(p Vec3) Vec3 {
	N := p.Sub(s.Center)
	N = N.Mul(1 / N.Len())
	return N
}

func (s Sphere) GetSpecular() float64 {
	return s.specular
}

func (s Sphere) GetClarity() float64 {
	return s.clarity
}

func (s Sphere) GetType() ObjectType {
	return SphereType
}

func NewSphere(color Color, center Vec3, radius, specular, clarity float64) Sphere {
	return Sphere{center, radius, radius * radius, specular, color, clarity}
}

type Triangle struct {
	A, B, C Vec3

	color    Color
	specular float64
	clarity  float64
}

func (t Triangle) GetColor() Color {
	return t.color
}

func (t Triangle) GetNorm() Vec3 {
	bToA := t.A.Sub(t.B)
	bToC := t.C.Sub(t.B)
	N := bToC.Cross(bToA)
	N = N.Mul(1 / N.Len())
	return N
}

func (t Triangle) GetSpecular() float64 {
	return t.specular
}

func (t Triangle) GetClarity() float64 {
	return t.clarity
}

func (t Triangle) GetType() ObjectType {
	return TriangleType
}

func (t Triangle) Rotate(axis Vec3, angle float64) Triangle {
	k := axis.Norm()

	cosAngle := math.Cos(angle)
	sinAngle := math.Sin(angle)

	var resA, resB, resC Vec3
	{
		one := t.A.Mul(cosAngle)
		two := k.Cross(t.A).Mul(sinAngle)
		three := k.Mul(k.Dot(t.A) * (1 - cosAngle))
		resA = one.Add(two).Add(three)
	}
	{
		one := t.B.Mul(cosAngle)
		two := k.Cross(t.B).Mul(sinAngle)
		three := k.Mul(k.Dot(t.B) * (1 - cosAngle))
		resB = one.Add(two).Add(three)
	}
	{
		one := t.C.Mul(cosAngle)
		two := k.Cross(t.C).Mul(sinAngle)
		three := k.Mul(k.Dot(t.C) * (1 - cosAngle))
		resC = one.Add(two).Add(three)
	}

	return Triangle{
		A: resA,
		B: resB,
		C: resC,

		color:    NewColor(0, 0, 255),
		specular: t.GetSpecular(),
		clarity:  t.GetClarity(),
	}
}

var spheres []Sphere = []Sphere{
	NewSphere(NewColor(255, 0, 0), NewVector(0, -1, 3), 1, 500, 0),
	NewSphere(NewColor(0, 255, 0), NewVector(2, 0, 4), 1, 10, 0),
	NewSphere(NewColor(0, 0, 255), NewVector(-2, 0, 4), 1, 500, 0.1),
	NewSphere(NewColor(255, 255, 0), NewVector(0, -5001, 0), 5000, 1000, 0),
}

var triangles []Triangle = []Triangle{
	{
		NewVector(2.9, -1, 6.8),  // A
		NewVector(-2.9, -1, 6.1), // B
		NewVector(0, 1.8, 5),     //C
		NewColor(255, 255, 255),
		10,
		0.5,
	},
}

type LightType int

const (
	AmbientLight LightType = iota
	PointLight
	DirectionLight
)

type Light struct {
	Type      LightType
	Position  *Vec3
	Intensity float64
}

func NewAmbientLight(intensity float64) Light {
	return Light{AmbientLight, nil, intensity}
}

func NewPointLight(position Vec3, intensity float64) Light {
	return Light{PointLight, &position, intensity}
}

func NewDirectionLight(position Vec3, intensity float64) Light {
	return Light{DirectionLight, &position, intensity}
}

var lights []Light = []Light{
	NewAmbientLight(0.2),
	NewPointLight(NewVector(2, 1, 1), 0.6),
	NewDirectionLight(NewVector(1, 4, 4), 0.2),
}

var cameraPosition Vec3 = NewVector(0, 0, 0)

const VW float64 = 1
const VH float64 = 1
const DISTANCE float64 = 1

var xDeg float64 = 0
var yDeg float64 = 0

func toRad(v float64) float64 {
	return v * (math.Pi / 180)
}

type Matrix [3][3]float64

func getDirection(xDeg, yDeg float64, v [3]float64) Vec3 {
	xRad := toRad(xDeg)
	yRad := toRad(yDeg)
	var mX Matrix = Matrix{
		{1, 0, 0},
		{0, math.Cos(xRad), -1 * math.Sin(xRad)},
		{0, math.Sin(xRad), math.Cos(xRad)},
	}
	var mY Matrix = Matrix{
		{math.Cos(yRad), 0, -1 * math.Sin(yRad)},
		{0, 1, 0},
		{math.Sin(yRad), 0, math.Cos(yRad)},
	}

	var res [3]float64 = [3]float64{0, 0, 0}
	M := mulMatrix(mY, mX)
	for i := range 3 {
		for j := range 3 {
			res[i] += v[j] * M[i][j]
		}
	}

	return NewVector(res[0], res[1], res[2])
}

func mulMatrix(m1, m2 Matrix) Matrix {
	var res Matrix
	n := len(m1)
	for i := range n {
		for j := range n {
			v1 := m1[i][0] * m2[0][j]
			v2 := m1[i][1] * m2[1][j]
			v3 := m1[i][2] * m2[2][j]
			res[i][j] = v1 + v2 + v3
		}
	}
	return res
}

func canvasToViewPort(x, y float64) Vec3 {
	vx := x * VW / float64(WIDTH)
	vy := y * VH / float64(HEIGHT)
	vz := DISTANCE
	D := NewVector(vx, vy, vz)
	return D
}

func intersectToSphere(origin Vec3, direction Vec3, s Sphere) (float64, float64) {
	OC := origin.Sub(s.Center)

	a := *direction.Magn
	b := 2 * OC.Dot(direction)
	c := OC.Dot(OC) - s.RadiusSqr

	discriminant := b*b - 4*a*c
	if discriminant < 0 {
		return math.Inf(1), math.Inf(1)
	}

	t1 := (-b + math.Sqrt(discriminant)) / (2 * a)
	t2 := (-b - math.Sqrt(discriminant)) / (2 * a)

	return (t1), (t2)
}

func intersectToTriangle(origin Vec3, direction Vec3, tri Triangle) float64 {
	// Find the normal of triangle's plane
	norm := tri.GetNorm()

	// Find if we intersect with the plane
	denom := norm.Dot(direction)
	if math.Abs(denom) < 0.001 {
		return math.Inf(1) // No solution
	}
	anom := norm.Dot(tri.A.Sub(origin))
	t := anom / denom

	P := origin.Add(direction.Mul(t))
	// If so then, check if we intersect within that triangle
	{
		aToB := tri.B.Sub(tri.A)
		bToC := tri.C.Sub(tri.B)
		cToA := tri.A.Sub(tri.C)

		aToPoint := P.Sub(tri.A)
		bToPoint := P.Sub(tri.B)
		cToPoint := P.Sub(tri.C)

		aTestVec := aToB.Cross(aToPoint)
		bTestVec := bToC.Cross(bToPoint)
		cTestVec := cToA.Cross(cToPoint)

		aValid := aTestVec.Dot(norm) > 0
		bValid := bTestVec.Dot(norm) > 0
		cValid := cTestVec.Dot(norm) > 0

		if aValid && bValid && cValid {
			return t
		}
		return math.Inf(1)
	}
}

func calculateReflection(L, N Vec3) Vec3 {
	// R = 2 * N * N.Dot(L) - L
	nDotL := N.Dot(L)
	R := N.Mul(2 * nDotL).Sub(L)
	return R
}

func computeIntensity(P, N, V Vec3, s float64) float64 {
	nLen := N.Len()
	vLen := V.Len()

	var intensity float64 = 0
	for _, l := range lights {
		if l.Type == AmbientLight {
			intensity += l.Intensity
			continue
		}

		var vecL Vec3
		var tMax float64

		if l.Type == PointLight {
			vecL = l.Position.Sub(P)
			tMax = 1.0
		} else {
			vecL = *l.Position
			tMax = math.Inf(1)
		}

		if isObstructedByObject(P, vecL, EPS, tMax) {
			continue
		}

		// Diffuse
		nDotL := N.Dot(vecL)
		if nDotL > 0 {
			intensity += l.Intensity * nDotL / (nLen * vecL.Len())
		}
		// Specular
		if s != -1 {
			R := calculateReflection(vecL, N)

			// cos(0) = (R.V / |R||V|)^s
			rDotV := R.Dot(V)
			if rDotV > 0 {
				cos := rDotV / (R.Len() * vLen)
				intensity += l.Intensity * math.Pow(cos, s)
			}
		}

	}
	return intensity
}

func isObstructedByObject(origin, direction Vec3, minT, maxT float64) bool {
	ok, _, _ := nearestObject(origin, direction, minT, maxT)
	return ok
}

func isValidT(t, tClosest, minT, maxT float64) bool {
	return minT < t && t < maxT && t < tClosest
}

func nearestObject(origin, direction Vec3, minT, maxT float64) (bool, Object, float64) {
	var selectedObject any = nil
	var tClosest float64 = math.Inf(1)

	for _, sphere := range spheres {
		t1, t2 := intersectToSphere(origin, direction, sphere)
		if isValidT(t1, tClosest, minT, maxT) {
			tClosest = t1
			selectedObject = sphere
		}
		if isValidT(t2, tClosest, minT, maxT) {
			tClosest = t2
			selectedObject = sphere
		}
	}

	for _, triangle := range triangles {
		t := intersectToTriangle(origin, direction, triangle)
		if isValidT(t, tClosest, minT, maxT) {
			tClosest = t
			selectedObject = triangle
		}
	}

	obj, ok := selectedObject.(Object)

	if ok && selectedObject != nil {
		return true, obj, tClosest
	}
	return false, nil, tClosest

}

func traceRay(origin Vec3, direction Vec3, tMin, tMax float64, depth int32) Color {
	ok, object, tClosest := nearestObject(origin, direction, tMin, tMax)
	if !ok {
		return Color{0, 0, 0}
	}

	P := origin.Add(direction.Mul(tClosest))

	var N Vec3
	if isSphere(object) {
		sph := object.(Sphere)
		N = sph.GetNorm(P)
	} else if isTriangle(object) {
		tri := object.(Triangle)
		N = tri.GetNorm()
	} else {
		assert(false, "Unexpected in traceRay")
	}

	V := direction.Mul(-1)
	intensity := computeIntensity(P, N, V, object.GetSpecular())

	currentColor := object.GetColor().Scale(intensity)
	if depth <= 0 || object.GetClarity() <= 0 {
		return currentColor
	}

	inverted := calculateReflection(direction.Mul(-1), N)
	reflectedColor := traceRay(P, inverted, EPS, math.Inf(1), depth-1)
	clarity := object.GetClarity()
	currentColor.Scale(clarity)
	reflectedColor.Scale(clarity)
	return currentColor.Add(reflectedColor)
}

func putPixel(x int32, y int32, color Color) {
	sx := WIDTH/2 + x
	sy := HEIGHT/2 - y - 1
	rl.DrawPixel(sx, sy, color.toRlColor())
}

type PixelJob struct {
	X, Y  int32
	Color Color
}

func Render() {
	jobs := make(chan PixelJob)

	for x := -(WIDTH / 2); x < (WIDTH / 2); x++ {
		go func() {
			for y := -(HEIGHT / 2); y < (HEIGHT / 2); y++ {
				D := canvasToViewPort(float64(x), float64(y))
				cameraDirection := getDirection(xDeg, yDeg, D.toArray())
				color := traceRay(cameraPosition, cameraDirection, 1, math.Inf(1), 2)
				jobs <- PixelJob{x, y, color}
			}
		}()
	}

	for x := -(WIDTH / 2); x < (WIDTH / 2); x++ {
		for y := -(HEIGHT / 2); y < (HEIGHT / 2); y++ {
			res := <-jobs
			putPixel(res.X, res.Y, res.Color)
		}
	}
}

func ToRight() {
	yDeg -= 10
}

func ToLeft() {
	yDeg += 10
}

func ToUp() {
	xDeg -= 10
}

func ToDown() {
	xDeg += 10
}

func main() {
	rl.InitWindow(WIDTH, HEIGHT, "Learn 3d")
	defer rl.CloseWindow()

	rl.SetTargetFPS(60)

	for !rl.WindowShouldClose() {
		d := rl.GetMouseDelta()
		if rl.IsMouseButtonDown(rl.MouseButtonLeft) && (d.X != 0 || d.Y != 0) {
			if d.X < 0 {
				ToRight()
			}
			if d.X > 0 {
				ToLeft()
			}
			if d.Y < 0 {
				ToDown()
			}
			if d.Y > 0 {
				ToUp()
			}
		}
		rl.BeginDrawing()
		Render()
		rl.EndDrawing()
	}
}
