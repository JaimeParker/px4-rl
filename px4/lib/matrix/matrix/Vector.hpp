/**
 * @file Vector.hpp
 *
 * Vector class.
 *
 * @author James Goppert <james.goppert@gmail.com>
 */

#pragma once

#include "Matrix.hpp"

namespace matrix
{

template<typename Type, size_t M>
class Vector : public Matrix<Type, M, 1>
{
public:
	using MatrixM1 = Matrix<Type, M, 1>;

	Vector() = default;

	PX4RL_HD Vector(const MatrixM1 &other) :
		MatrixM1(other)
	{
	}

	PX4RL_HD explicit Vector(const Type data_[M]) :
		MatrixM1(data_)
	{
	}

	template<size_t P, size_t Q>
	PX4RL_HD Vector(const Slice<Type, M, 1, P, Q> &slice_in) :
		Matrix<Type, M, 1>(slice_in)
	{
	}

	template<size_t P, size_t Q, size_t DUMMY = 1>
	PX4RL_HD Vector(const Slice<Type, 1, M, P, Q> &slice_in)
	{
		Vector &self(*this);

		for (size_t i = 0; i < M; i++) {
			self(i) = slice_in(0, i);
		}
	}

	template<size_t P, size_t Q>
	PX4RL_HD Vector(const ConstSlice<Type, M, 1, P, Q> &slice_in) :
		Matrix<Type, M, 1>(slice_in)
	{
	}

	template<size_t P, size_t Q, size_t DUMMY = 1>
	PX4RL_HD Vector(const ConstSlice<Type, 1, M, P, Q> &slice_in)
	{
		Vector &self(*this);

		for (size_t i = 0; i < M; i++) {
			self(i) = slice_in(0, i);
		}
	}

	PX4RL_HD inline const Type &operator()(size_t i) const
	{
		assert(i < M);

		const MatrixM1 &v = *this;
		return v(i, 0);
	}

	PX4RL_HD inline Type &operator()(size_t i)
	{
		assert(i < M);

		MatrixM1 &v = *this;
		return v(i, 0);
	}

	PX4RL_HD Type dot(const MatrixM1 &b) const
	{
		const Vector &a(*this);
		Type r(0);

		for (size_t i = 0; i < M; i++) {
			r += a(i) * b(i, 0);
		}

		return r;
	}

	PX4RL_HD inline Type operator*(const MatrixM1 &b) const
	{
		const Vector &a(*this);
		return a.dot(b);
	}

	PX4RL_HD inline Vector operator*(Type b) const
	{
		return Vector(MatrixM1::operator*(b));
	}

	PX4RL_HD Type norm() const
	{
		const Vector &a(*this);
		return Type(std::sqrt(a.dot(a)));
	}

	PX4RL_HD Type norm_squared() const
	{
		const Vector &a(*this);
		return a.dot(a);
	}

	PX4RL_HD inline Type length() const
	{
		return norm();
	}

	PX4RL_HD inline void normalize()
	{
		(*this) /= norm();
	}

	PX4RL_HD Vector unit() const
	{
		return (*this) / norm();
	}

	PX4RL_HD Vector unit_or_zero(const Type eps = Type(1e-5)) const
	{
		const Type n = norm();

		if (n > eps) {
			return (*this) / n;
		}

		return Vector();
	}

	PX4RL_HD inline Vector normalized() const
	{
		return unit();
	}

	PX4RL_HD bool longerThan(Type testVal) const
	{
		return norm_squared() > testVal * testVal;
	}

	PX4RL_HD Vector sqrt() const
	{
		const Vector &a(*this);
		Vector r;

		for (size_t i = 0; i < M; i++) {
			r(i) = Type(std::sqrt(a(i)));
		}

		return r;
	}

	PX4RL_HD void print() const
	{
		(*this).transpose().print();
	}

	PX4RL_HD static size_t size()
	{
		return M;
	}
};

template<typename OStream, typename Type, size_t M>
PX4RL_HD OStream &operator<<(OStream &os, const matrix::Vector<Type, M> &vector)
{
	os << "\n";
	// element: tab, point, 8 digits, 4 scientific notation chars; row: newline; string: \0 end
	static const size_t n = 15 * M * 1 + 1 + 1;
	char string[n];
	vector.transpose().write_string(string, n);
	os << string;
	return os;
}

} // namespace matrix
