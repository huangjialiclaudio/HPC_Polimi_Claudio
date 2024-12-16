/*
  This code wants to be an exampple of the use of const methods.
  Here we create a getter/setter member function whith the same name 
  but different constness and return type. We mensiont the the same 
  technique is used for the addressing operator [] in a comtainer-type
  class.
*/ 

#include <iostream>
class Foo
{
public:
  // The const version returns a value
  double b()const
    {
      std::cout<<"calling the const version\n";
      std::cout<<"The internal variable value is "<<b_<<std::endl;
      return b_;
    }
  // The non const verson returns a reference
  double & b()
    {
      std::cout<<"calling the non-const version\n";
      std::cout<<"The internal variable value is "<<b_<<std::endl;
      return b_;
    }
private:
  double b_=10.;// default value
};
  

int main()
{
  Foo foo;
  const Foo constFoo;
  double z;
  std::cout<<"z = foo.b()\n";
  z = foo.b();
  std::cout<<"z="<<z<<std::endl;
  std::cout<<"z = constFoo.b()\n";
  z=constFoo.b();
  std::cout<<"z="<<z<<std::endl;
  std::cout<<"foo.b()=90.\n";
  foo.b()=90.;
  //std::cout<<"constFoo.b()=90.\n";
  //constFoo.b()=90.; //COMPILER ERROR!!
}
/*
In my compiler the compiler error you obtain by uncommenting the last two lines
of the code is 

main.cpp:38:13: error: lvalue required as left operand of assignment
   38 |   constFoo.b()=90.; //COMPILER ERROR!!
      |   ~~~~~~~~~~^~

Myaybe you find th eerror a bit criptic. To undertand what it means I
recall that a lvalue is a value category, and is THE ONLY category of
values that can stay at the left side of an assigment.

The category of a value returned by a function (when the function
returns a value, not a reference) is *not lvalue* (is an
rvalue). Therefore the value returned by a function CANNOT stay at the
left hand side of an assignment.

 */
