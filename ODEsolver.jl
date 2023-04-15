using LinearAlgebra

mutable struct ODE_initial
    x_start::Int
    h::Float64
    x_end::Int
    Λ::Function

    function ODE_initial(x_start::Int, h::Float64, x_end::Int, Λ::Function) 
    if x_start < 0 
        throw(ArgumentError("initial age must be positive"))
    end
    if h < 0 
        throw(ArgumentError("the stepsize h must be positive"))
    end
    if x_end < 0 || x_end < x_start
        throw(ArgumentError("Please choose last age > intial age, and keep it positive"))
    end
    new(x_start, h, x_end, Λ)
    end
end

function inital_array(p::ODE_initial)
    x_start = p.x_start
    h = p.h
    x_end = p.x_end
    Λ = p.Λ

    d = size(Λ(0))[1]

    P0 = Matrix(1.0*I, d,d)

    if x_start == x_end 
        return P0
    end

    N = Int((x_end-x_start)/h)


    P = zeros(Float64, d, d, N+1)
    P[:,:,1] = P0

    return P
end

# Forward Kolmogorov
function f(t,M, Λ)
    return M*Λ(t)
end

function Euler(p::ODE_initial)
    x_start = p.x_start
    x_end = p.x_end
    h = p.h
    Λ = p.Λ

    #Number of steps:
    N = Int((x_end-x_start)/h)

    P = inital_array(p)

    for n in 1:N
        P[:,:, n+1] = P[:,:,n] + h*f(x_start+n*h, P[:,:,n], Λ)
    end

    return P
end

function Taylor(p::ODE_initial)
    x_start = p.x_start
    x_end = p.x_end
    h = p.h
    Λ = p.Λ

    #Number of steps:
    N = Int((x_end-x_start)/h)

    P = inital_array(p)
    Id = P[:,:,1]

    for n in 1:N
        P[:,:, n+1] = P[:,:,n]*(Id + (h/2)*Λ(x_start + n*h) + (h/2)*Λ(x_start + n*h + h) + (h^(2)/2)*(Λ(x_start+n*h))^2)
    end

    return P
end

function RK4(p::ODE_initial)
    x_start = p.x_start
    x_end = p.x_end
    h = p.h
    Λ = p.Λ

    #Number of steps:
    N = Int((x_end-x_start)/h)

    P = inital_array(p)

    function k1(t,M, Λ)
        return f(t,M, Λ)
    end
    
    function k2(t,M, Λ)
        return f(t+h/2, M +h*k1(t,M,Λ)/2, Λ)
    end
    
    function k3(t,M, Λ)
        return f(t+h/2, M+ h*k2(t, M, Λ)/2, Λ)
    end
    
    function k4(t,M, Λ)
        return f(t+h, M + h*k3(t,M, Λ), Λ)
    end

    for n in 1:N
        P[:,:,n+1] = P[:,:,n] + (h/6)*(k1(x_start + n*h, P[:,:,n], Λ) + 2*k2(x_start +n*h, P[:,:,n], Λ) +
                                       2*k3(x_start +n*h, P[:,:, n], Λ) + k4(x_start +n*h, P[:,:,n], Λ))
    end

    return P
end





