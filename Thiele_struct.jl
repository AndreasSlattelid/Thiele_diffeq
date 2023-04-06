mutable struct Contract
    age::Real
    contract_length::Real
    mortality_rates::Function
    step_size::Real
    
    function Contract(age::Real, contract_length::Real, mortality_rates::Function, step_size::Real)
        if age < 0
            throw(ArgumentError("Age cannot be negative"))
        end
        if contract_length < 0
            throw(ArgumentError("Contract length cannot be negative"))
        end
        if step_size < 0
            throw(ArgumentError("Step size cannot be negative"))
        end
        new(age, contract_length, mortality_rates, step_size)
    end
end

function disability_benefit(a::Contract, D::Int64, B::Int64,r::Float64)
    "
    Args:
        a (Contract): Inherits from Contract 
        D (Int64): The yearly disability paymnent 
        B (Int64): The Death Benefit 
        r (Float64): the constant annual interest rate: 
    Returns: 
        Array: (V_{*}(t), V_{⋄}(t), π0, π)
        V_{*}(t): Reserve for healty state 
        V_{⋄}(t): Reserve for disabled state 
        π0: One time premium 
        π: yearly_premium
    "
    
    x = a.age
    h = a.step_size
    T = a.contract_length
    N = floor(Int, T/h)

    #S = {*, ⋄, †} = {0,1,2} 
    n_states = 3
    Λ = a.mortality_rates
    size(Λ(0)) == (n_states,n_states) || error("Λ must be $n_states x $n_states")

    #mortality rates for V_{*}^{+}(t, A)    
    μ01(t) = Λ(t)[1,2]
    μ02(t) = Λ(t)[1,3]

    #motality rates for V_{⋄}(t,A):
    μ10(t) = Λ(t)[2,1]
    μ12(t) = Λ(t)[2,3]
    
    #policy function
    function a_dis(t)
        if t>= 0 && t < T
            ans = D
        else 
            ans = 0 
        end
        return ans
    end

    V_act = zeros(N+1)
    V_dis = zeros(N+1)

    #CASE 1: we do not pay premiums
    #= 
    d/dt V_{*}(t) = r(t)V_{*}(t) - μ01(x+t)[V_{⋄}(t)-V_{*}(t)]
                                 - μ02(x+t)[B⋅I_[0,T)(t)-V_{*}(t)]
    
    d/dt V_{⋄}(t) = r(t)V_{⋄}(t) - D⋅I_[0,T)(t) - μ01(x+t)[V_{*}(t)-V_{⋄}(t)] + μ12(x+t)V_{⋄}(t)
    =#

    for i in reverse(1:N)
        tᵢ = (i+1)*h
        V_act[i] = V_act[i+1] .- h.*(r.*V_act[i+1] 
                                   .- μ01(x + tᵢ).*(V_dis[i+1] .- V_act[i+1])
                                   .- μ02(x + tᵢ).*(B .- V_act[i+1])
                                   )
        
        V_dis[i] = V_dis[i+1] .-h.*(r.*V_dis[i+1] - a_dis(tᵢ)
                                    .- μ10(x + tᵢ).*(V_act[i+1]-V_dis[i+1])
                                    .+ μ12(x + tᵢ).*V_dis[i+1]
                                    )
        
    end

    #One time premium
    π0 = V_act[1]

    #CASE 2: we pay premium of 1 NOK per year:

    #The a.e derivative for the case where we pay 1NOK premium per year: 
    function a_prem1(t)
        if t >= 0 && t < T
            ans = -1 
        else 
            ans = 0
        end
        return ans 
    end

    V_act_p1 = zeros(N+1)
    V_dis_p1 = zeros(N+1)

    for i in reverse(1:N)
        tᵢ = (i+1)*h
        V_act_p1[i] = V_act_p1[i+1] .-h.*(r.*V_act_p1[i+1] - a_prem1(tᵢ)
                                          .- μ01(x + tᵢ).*(V_dis_p1[i+1].-V_act_p1[i+1])
                                          .- μ02(x + tᵢ).*V_dis_p1[i+1]
                                          )
        V_dis_p1[i] = V_dis_p1[i+1] .-h.*(r*V_dis_p1[i+1] 
                                          .- μ10(x + tᵢ).*(V_act_p1[i+1]-V_dis_p1[i+1])
                                          .+ μ12(x + tᵢ).*V_dis_p1[i+1]
                                          )
    end

    π = -(π0/V_act_p1[1])

    return (V_act, V_dis, π0, π)
end


function endownment(a::Contract, E::Int64, B::Int64,r::Float64)
    "
    Args:
        a (Contract): Inherits from Contract 
        E (Int64): The yearly endownmnet 
        B (Int64): The Death Benefit 
        r (Float64): the constant annual interest rate: 
    Returns: 
        Array: (V_{*}(t), π0, π)
        V_{*}(t): Reserve for healty state 
        π0: One time premium 
        π: yearly_premium
    "
    x = a.age
    h = a.step_size
    T = a.contract_length
    N = floor(Int, T/h)

    #S = {*, †} = {0,1} 
    n_states = 2
    Λ = a.mortality_rates
    size(Λ(0)) == (n_states,n_states) || error("Λ must be $n_states x $n_states")

    #mortality rates for V_{*}^{+}(t, A)    
    μ01(t) = Λ(t)[1,2]

    V_act = zeros(N+1) 
    V_act[N+1] = E

    #= 
    d/dt V_{*}(t) = rV_{*}(t) - μ01(x+t)[B - V_{*}(t)]
    =#

    for i in reverse(1:N)
        tᵢ = (i+1)*h
        V_act[i] = V_act[i+1] .- h.*(r.*V_act[i+1] 
                                     .-μ01(x + tᵢ).*(B.-V_act[i+1])
                                     )                             
    end

    #One time premium
    π0 = V_act[1]

    #CASE 2: we pay premium of 1 NOK per year:

    #The a.e derivative for the case where we pay 1NOK premium per year: 
    function a_prem1(t)
        if t >= 0 && t < T
            ans = -1 
        else 
            ans = 0
        end
        return ans 
    end

    V_act_p1 = zeros(N+1)
    
    for i in reverse(1:N)
        tᵢ = (i+1)*h
        V_act_p1[i] = V_act_p1[i+1] .- h.*(r.*V_act_p1[i+1] .- a_prem1(tᵢ)
                                     .+ μ01(x + tᵢ).*V_act_p1[i+1]
                                     )
    end
    V_act_p1[1]

    π = -(π0/V_act_p1[1])
    return (V_act, π0, π)
end

#EXAMPLES
#--------------------------------------------------------------------------------------------------------
#example of disability:
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

contract_dis = Contract(60, 10, Λ_dis, 1/12)

disability_benefit(contract_dis, 25_000, 60_000, 0.04)

#example of endownment:
function Λ_endownment(t)
    μ01(t) = 0.0015 + 0.0004*(t-35)
    μ00(t) = -μ01(t)

    L = [μ00(t) μ01(t)
         0        0  ]

    return L
end

contract_endownment = Contract(35, 60, Λ_endownment, 1/12)

endownment(contract_endownment,125_000, 250_000, 0.035)

#--------------------------------------------------------------------------------------------------------