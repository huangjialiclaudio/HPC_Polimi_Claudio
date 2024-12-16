#include <array>
#include <vector>
#include <numbers>
#include <memory>
#include <chrono>
#include <iostream>

using namespace std;

class Shape {
public:
    // constructor that initializes the name
    Shape() = default;
    // pure virtual getter for the area
    virtual double getArea() const = 0;
    // getter for the shape name (non virtual)
    constexpr virtual const char *getName() = 0;
    // virtual destructor
    virtual ~Shape() = default;
private:
    // member with the name of the shape (const)
};

// Implement the classes "Circle" and "Rectangle"
// The constructor must be empty, in the sense the name of the shape
// should not be a user's choice 
class Circle : public Shape{
public:
    // constructor that initializes the name
    Circle(double radius) : Shape() , radius(radius){};
    // pure virtual getter for the area
    virtual double getArea() const override {return radius * radius * 3.14;};
    // getter for the shape name (non virtual)
    constexpr virtual const char *getName() override {return "Circle";};
    // virtual destructor
    virtual ~Circle() override = default;
private:
    const double radius;
};

class Rectangular : public Shape{
public:
    // constructor that initializes the name
    Rectangular(double basis, double height) : Shape() , basis(basis) , height(height){};
    // pure virtual getter for the area
    virtual double getArea() const override {return basis * height;};
    // getter for the shape name (non virtual)
    constexpr virtual const char *getName() override {return "Rectangular";};
    // virtual destructor
    virtual ~Rectangular() override = default;
private:
    const double basis, height;
};


int main() {
    // Instantiate vector of shapes
    std::vector<std::shared_ptr<Shape>> shapes;
    shapes.push_back(std::make_shared<Circle>(1.0));
    shapes.push_back(std::make_shared<Rectangular>(4.0,3.0));
    // Add some shapes
    
    // Loop over shapes and print
    for (const auto& s : shapes) {
        std::cout << "I am a " << s->getName() << ", my area is: " << s->getArea() << std::endl;
    }
    return 0;
}