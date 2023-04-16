using LinearAlgebra

mutable struct ODE_initial
    x_start::Int
    x_end::Int
    h::Float64
    Λ::Function

    function ODE_initial(x_start::Int, x_end::Int, h::Float64, Λ::Function) 
    if x_start < 0 
        throw(ArgumentError("initial age must be positive"))
    end
    if x_end < 0 || x_end < x_start
        throw(ArgumentError("Please choose last age > intial age, and keep it positive"))
    end
    if h < 0 
        throw(ArgumentError("the stepsize h must be positive"))
    end
    new(x_start, x_end, h, Λ)
    end
end

function inital_array(p::ODE_initial)
    " 
    Returns the inital array P = (I, 0*I, 0*I, ...)
    I = [1 0 0
         0 1 0
         0 0 1]
    I is d×d
    "
    x_start = p.x_start
    h = p.h
    x_end = p.x_end
    Λ = p.Λ

    #dimension of Λ and thus number of states:
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
    "
    d/dt P(s,t) = P(s,t)Λ(t)
    function of three arguments as it could not take Λ(t) as arg
    "
    return M*Λ(t)
end

function Euler(p::ODE_initial)
    "
    P(s, t+h) = P(s,t) + hP(s,t)Λ(t)
    Returns: 
        P: (Array{Float64}), contains P(x_start, x_start), \dots P(x_start, x_end)
    "
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
    "
    P(s, t+h) = P(s,t)*[I+hΛ(t) + (h^{2}/2)⋅Λ'(t) + (h^{2}/2)Λ(t)^(2)] + O(h^{3})
    Λ'(t) ≈ [Λ(t+h)-Λ(t)]/h
    Returns: 
        P: (Array{Float64}), contains P(x_start, x_start), \dots P(x_start, x_end)
    "
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

#Disability:
function Λ_dis(t)
    #state0:
    μ01(t) = 0.0004 + 10^(0.06*t-5.46)
    μ02(t) = 0.0005 + 10^(0.038*t-4.12)
    μ00(t) = -(μ01(t) + μ02(t))
    #state1:
    μ10(t) = 0.05
    μ12(t) = μ02(t)
    μ11(t) = -(μ10(t)+μ12(t))
    #state2:
    # transition rates in the deceased state are zero

    L = [μ00(t) μ01(t) μ02(t)
         μ10(t) μ11(t) μ12(t)
         0       0     0    ]

    return L
end

test = ODE_initial(25, 30, 1/12, Λ_dis)

Taylor(test)


