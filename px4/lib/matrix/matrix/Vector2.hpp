/**
 * @file Vector2.hpp
 *
 * 2D vector class.
 *
 * @author James Goppert <james.goppert@gmail.com>
 */

#pragma once

#include "Vector.hpp"

namespace matrix
{

template<typename Type>
class Vector2 : public Vector<Type, 2>
{
public:

	using Matrix21 = Matrix<Type, 2, 1>;
	using Vector3 = Vector<Type, 3>;

	Vector2() = default;

	PX4RL_HD Vector2(const Matrix21 &other) :
		Vector<Type, 2>(other)
	{
	}

	PX4RL_HD explicit Vector2(const Type data_[2]) :
		Vector<Type, 2>(data_)
	{
	}

	PX4RL_HD Vector2(Type x, Type y)
	{
		Vector2 &v(*this);
		v(0) = x;
		v(1) = y;
	}

	using base = Vector<Type, 2>;
	using base::base;

	PX4RL_HD explicit Vector2(const Vector3 &other)
	{
		Vector2 &v(*this);
		v(0) = other(0);
		v(1) = other(1);
	}

	PX4RL_HD Type cross(const Matrix21 &b) const
	{
		const Vector2 &a(*this);
		return a(0) * b(1, 0) - a(1) * b(0, 0);
	}

	/**
	 * Override matrix ops so Vector2 type is returned
	 */

	PX4RL_HD Vector2 operator+(Vector2 other) const
	{
		return Matrix21::operator+(other);
	}

	PX4RL_HD Vector2 operator+(Type scalar) const
	{
		return Matrix21::operator+(scalar);
	}

	PX4RL_HD Vector2 operator-(Vector2 other) const
	{
		return Matrix21::operator-(other);
	}

	PX4RL_HD Vector2 operator-(Type scalar) const
	{
		return Matrix21::operator-(scalar);
	}

	PX4RL_HD Vector2 operator-() const
	{
		return Matrix21::operator-();
	}

	PX4RL_HD Vector2 operator*(Type scalar) const
	{
		return Matrix21::operator*(scalar);
	}

	PX4RL_HD Type operator*(Vector2 b) const
	{
		return Vector<Type, 2>::operator*(b);
	}

	PX4RL_HD Type operator%(const Matrix21 &b) const
	{
		return (*this).cross(b);
	}

};

using Vector2f = Vector2<float>;
using Vector2d = Vector2<double>;

} // namespace matrix
