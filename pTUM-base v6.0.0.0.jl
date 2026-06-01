# pTUM-base v6.0.0.0

function initCatDist(nvars,ncats,layers;shift=0.1)
  L=length(layers)+1
  maxWidth=(length(layers)==0 ? 0 : maximum(layers))
  w=zeros(L,max(ncats-1,maxWidth),max(nvars-1,maxWidth))
  b=zeros(L,max(ncats-1,maxWidth))
  npars=0
  for l=1:L
    J=(l==1 ? nvars-1 : layers[l-1])
    I=(l==L ? ncats-1 : layers[l])
    for i=1:I
      b[l,i]=(l==L ? 0 : atanh((2*i-I-1)/I)+shift)
      npars+=J+1
    end

  end
  return(w,b,npars)
end

function initConDist(nvars,layers1,layers2;shift=0.1)
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  m1=max(nvars-(l1==0 ? 0 : 1),(l1>0 ? layers1[l1]+1 : 1))
  m2=max((l1>0 ? maximum(layers1) : 1),(l2>0 ? maximum(layers2) : 1))
  w=zeros(L,m2,max(m1,m2))
  b=zeros(L,m2)
  npars=0
  for l=1:L
    J=(l==1 ? nvars-((l1==0) ? 0 : 1) : (l<=l1+1 ? (layers1[l-1]+(l==l1+1 ? 1 : 0)) : layers2[l-1-l1]))
    I=(l==L ? 1 : (l<=l1 ? layers1[l] : layers2[l-l1]))
    for i=1:I
      b[l,i]=(l==L ? 0 : atanh((2*i-I-1)/I)+shift)
      npars+=J+1
    end
  end
  return(w,b,npars)
end

function initTimDist(nvars,nevents,layers1,layers2;shift=0.1)
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  m1=max(nvars-(l1==0 ? 1 : 2),(l1>0 ? layers1[l1]+1 : 1),(l2>0 ? layers2[1]+nevents : 1))
  m2=max((l1>0 ? maximum(layers1) : 1),(l2>0 ? maximum(layers2) : 1))
  m3=max(m2,nevents+(l2>0 ? layers2[1] : nevents))
  w=zeros(L,m3,max(m1,m2))
  b=zeros(L,m3)
  npars=0
  for l=1:L
    J=(l==1 ? nvars-((l1==0) ? 1 : 2) : (l<=l1+1 ? (layers1[l-1]+(l==l1+1 ? 1 : 0)) : (layers2[l-1-l1]+(l==l1+2 ? nevents : 0))))
    I=(l==L ? nevents+((l2==0) ? nevents : 0) : (l<=l1 ? layers1[l] : layers2[l-l1]+(l==l1+1 ? nevents : 0)))
    for i=1:I
      b[l,i]=(((l==L)||((l==l1+1)&&(i<=nevents))) ? 0 : atanh((2*i-I-1-(l==l1+1 ? nevents : 0))/(I-(l==l1+1 ? nevents : 0)))+shift)
      npars+=1
      for j=1:J
        if !(((l==l1+1)&&(i<=nevents))||((l==l1+2)&&(j<=nevents)))
          w[l,i,j]=(((l>l1+1)||((l==l1+1)&&(j==J))) ? shift : 0)
          npars+=1
        end
      end
    end
  end
  return(w,b,npars)
end

function fitCatDist(w,b,data,categories,layers,iterations;hotvars=Int64[],obsweights=false,α=0.001,β1=0.9,β2=0.999,ϵ=1e-8,c=[3.0])
  L=length(layers)+1
  ncats=length(categories)
  N,nvars=size(data,1),size(data,2)-(obsweights ? 1 : 0)
  sw2,sw3,sb2=size(w,2),size(w,3),size(b,2)
  x,u,∂h∂x,s1=ntuple(_->zeros(L,sb2),4)
  cmax=length(c)
  llv=Float64[]
  # sum of observations per categorical value
  nhotvars=length(hotvars)
  if nhotvars>0
    hotsums=sum(data[:,1:sum(hotvars)] .* (obsweights ? data[:,nvars+1] : ones(N)),dims=1)
    hotvals=minimum(data[:,1:sum(hotvars)],dims=1)+maximum(data[:,1:sum(hotvars)],dims=1)
    nobs= (obsweights ? sum(data[:,nvars+1]) : N)
  end
  # Adam optimization
  mb,vb=ntuple(_->zeros(L,sb2),2)
  mw,vw=ntuple(_->zeros(L,sw2,sw3),2)
  println()
  println("Maximizing likelihood categorical distribution..")
  for iteration=0:iterations
    ∂h∂w,∂h∂b=zeros(L,sw2,sw3),zeros(L,sb2)
    h=0
    for n=1:N
      obs=data[n,:]
      obsw=(obsweights ? obs[nvars+1] : 1)
      catnr=ncats
      # propagate forward
      for l=1:L
        J = (l==1 ? nvars-1 : layers[l-1])
        I = (l==L ? ncats-1 : layers[l])
        for i=1:I
          u[l,i]=b[l,i]
          for j=1:J
            u[l,i]+=w[l,i,j]*(l==1 ? obs[j] : x[l-1,j])
          end
          x[l,i]=(l==L ? 1/(1+exp(-u[l,i])) : tanh(u[l,i]))
          s1[l,i]=(l==L ? x[l,i]*(1-x[l,i]) : 1-x[l,i]^2)
          if (l==L)
            if obs[nvars]==categories[i]
              catnr=i
            end
            ∂h∂x[l,i]=obsw*(i<catnr ? 1/(x[l,i]-1) : (i==catnr ? 1/x[l,i] : 0))
            h+=obsw*(i<catnr ? log(1-x[l,i]) : (i==catnr ? log(x[l,i]) : 0))
          end
        end
      end
      # propagate backward
      if iteration<iterations
        l=L-1
        while l>=0
          I=(l==0 ? nvars-1 : layers[l])
          J=(l==L-1 ? ncats-1 : layers[l+1])
          for i=1:I
            if (l>0)
              ∂h∂x[l,i]=0
            end
            for j=1:J
              if (l>0)
                ∂h∂x[l,i]+=w[l+1,j,i]*∂h∂x[l+1,j]*s1[l+1,j]
              end
              if i==1
                ∂h∂b[l+1,j]+=∂h∂x[l+1,j]*s1[l+1,j]
                if n==N
                  mb[l+1,j]=β1*mb[l+1,j]+(1-β1)*∂h∂b[l+1,j]
                  vb[l+1,j]=β2*vb[l+1,j]+(1-β2)*∂h∂b[l+1,j]^2
                  b[l+1,j]+=α*mb[l+1,j]/((1-β1^(iteration+1))*(sqrt(vb[l+1,j]/(1-β2^(iteration+1)))+ϵ))
                end
              end
              ∂h∂w[l+1,j,i]+=∂h∂x[l+1,j]*s1[l+1,j]*(l==0 ? obs[i] : x[l,i])
              if n==N
                mw[l+1,j,i]=β1*mw[l+1,j,i]+(1-β1)*∂h∂w[l+1,j,i]
                vw[l+1,j,i]=β2*vw[l+1,j,i]+(1-β2)*∂h∂w[l+1,j,i]^2
                w[l+1,j,i]=max(-c[min(2,cmax)],min(c[min(2,cmax)],w[l+1,j,i]+α*mw[l+1,j,i]/((1-β1^(iteration+1))*(sqrt(vw[l+1,j,i]/(1-β2^(iteration+1)))+ϵ))))
              end
            end
          end
          l-=1
        end
      end
    end
    # balance hot-encoded categorical variables
    if (nhotvars>0)&&(iteration<iterations)
      J=(L==1 ? ncats-1 : layers[1])
      for j=1:J
        i=1
        for hotvar=1:nhotvars
          imbalance=0
          hots=hotvars[hotvar]
          if hots==1
            i+=1
            break
          end
          ii=hots
          while ii>0
            imbalance+=w[1,j,i]*hotsums[i]
            ii-=1
            i+=1
          end
          imbalance=imbalance/nobs
          b[1,j]+=imbalance
          for ii=1:hots
            w[1,j,i-ii]=max(-c[1],min(c[1],w[1,j,i-ii]-(hotvals[i-ii]==0 ? 0 : imbalance/hotvals[i-ii])))
          end
        end
      end
    end
    push!(llv,h)
    if (mod(iteration,100)==0)||(iteration==iterations)  
      print("\e[2K")
      print("\e[1G")
      print("iteration ",iteration,"/",iterations,", log-likelihood=",h)
    end
  end
  println()
  return(w,b,llv)
end

function fitConDist(w,b,data,layers1,layers2,iterations;hotvars=Int64[],obsweights=false,α=0.001,β1=0.9,β2=0.999,ϵ=1e-8,c=[3.0])
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  N,nvars=size(data,1),size(data,2)-(obsweights ? 1 : 0)
  sw2,sw3,sb2=size(w,2),size(w,3),size(b,2)
  x,z,u,v,∂h∂x,∂h∂z,s1,s2=ntuple(_->zeros(L,sb2),8)
  cmax=length(c)
  llv=Float64[]
  # sum of observations per categorical value
  nhotvars=length(hotvars)
  if nhotvars>0
    hotsums=sum(data[:,1:sum(hotvars)].*(obsweights ? data[:,nvars+1] : ones(N)),dims=1)
    hotvals=minimum(data[:,1:sum(hotvars)],dims=1)+maximum(data[:,1:sum(hotvars)],dims=1)
    nobs= (obsweights ? sum(data[:,nvars+1]) : N)
  end
  # Adam optimization
  mb,vb=ntuple(_->zeros(L,sb2),2)
  mw,vw=ntuple(_->zeros(L,sw2,sw3),2)
  println()
  println("Maximizing likelihood continuous distribution..")
  for iteration=0:iterations
    ∂h∂w,∂h∂b=zeros(L,sw2,sw3),zeros(L,sb2)
    h=0
    for n=1:N
      obs=data[n,:]
      obsw=(obsweights ? obs[nvars+1] : 1)
      # propagate forward
      for l=1:L
        J = (l==1 ? nvars-((l1==0) ? 0 : 1) : (l<=l1+1 ? (layers1[l-1]+(l==l1+1 ? 1 : 0)) : layers2[l-1-l1]))
        I = (l==L ? 1 : (l<=l1 ? layers1[l] : layers2[l-l1]))
        for i=1:I
          u[l,i]=b[l,i]
          v[l,i]=0
          for j=1:J
            u[l,i]+=w[l,i,j]*(l==1 ? obs[j] : (((l==l1+1)&&(j==J)) ? obs[nvars] : x[l-1,j]))
            if l>=l1+1
              v[l,i]+=w[l,i,j]*(l==l1+1 ? (j==J ? 1 : 0) : z[l-1,j])
            end
          end
          x[l,i]=(l==L ? 1/(1+exp(-u[l,i])) : tanh(u[l,i]))
          s1[l,i]=(l==L ? x[l,i]*(1-x[l,i]) : 1-x[l,i]^2)
          s2[l,i]=(l==L ? (1-2*x[l,i])*s1[l,i] : -2*x[l,i]*s1[l,i])
          z[l,i]=(l<=l1 ? 0 : max(s1[l,i]*v[l,i],1e-18))
        end
      end
      h+=obsw*log(z[L,1])
      ∂h∂x[L,1]=0
      ∂h∂z[L,1]=obsw/z[L,1]
      # propagate backward
      if iteration<iterations
        l=L-1
        while l>=0
          I=(l==0 ? nvars-((l1==0) ? 0 : 1) : (l<=l1 ? (layers1[l]+(l==l1 ? 1 : 0)) : layers2[l-l1]))
          J=(l+1==L ? 1 : (l+1<=l1 ? layers1[l+1] : layers2[l+1-l1]))
          for i=1:I
            if (l>0)&&((i<I)||(l!=l1))
              ∂h∂x[l,i]=0
              ∂h∂z[l,i]=0
            end
            for j=1:J
              if (l>0)&&((i<I)||(l!=l1))
                ∂h∂x[l,i]+=w[l+1,j,i]*(∂h∂x[l+1,j]*s1[l+1,j]+∂h∂z[l+1,j]*s2[l+1,j]*v[l+1,j])
                ∂h∂z[l,i]+=w[l+1,j,i]*∂h∂z[l+1,j]*s1[l+1,j]
              end
              if i==1
                ∂h∂b[l+1,j]+=∂h∂x[l+1,j]*s1[l+1,j]+∂h∂z[l+1,j]*s2[l+1,j]*v[l+1,j]
                if n==N
                  mb[l+1,j]=β1*mb[l+1,j]+(1-β1)*∂h∂b[l+1,j]
                  vb[l+1,j]=β2*vb[l+1,j]+(1-β2)*∂h∂b[l+1,j]^2
                  b[l+1,j]+=α*mb[l+1,j]/((1-β1^(iteration+1))*(sqrt(vb[l+1,j]/(1-β2^(iteration+1)))+ϵ))
                end
              end
              ∂h∂w[l+1,j,i]+=∂h∂x[l+1,j]*s1[l+1,j]*(l==0 ? obs[i] : (((l==l1)&&(i==I)) ? obs[nvars] : x[l,i]))+∂h∂z[l+1,j]*
              (s2[l+1,j]*v[l+1,j]*(l==0 ? obs[i] : (((l==l1)&&(i==I)) ? obs[nvars] : x[l,i]))+s1[l+1,j]*(((l==l1)&&(i==I)) ? 1 : (l==0 ? 0 : z[l,i])))
              if n==N
                mw[l+1,j,i]=β1*mw[l+1,j,i]+(1-β1)*∂h∂w[l+1,j,i]
                vw[l+1,j,i]=β2*vw[l+1,j,i]+(1-β2)*∂h∂w[l+1,j,i]^2
                w[l+1,j,i]=max(-c[min(2,cmax)],min(c[min(2,cmax)],w[l+1,j,i]+α*mw[l+1,j,i]/((1-β1^(iteration+1))*(sqrt(vw[l+1,j,i]/(1-β2^(iteration+1)))+ϵ))))
                if l>l1 || ((l==l1)&&(i==I))
                  w[l+1,j,i]=max(w[l+1,j,i],0)
                end
              end
            end
          end
          l-=1
        end
      end
    end
    # balance hot-encoded categorical variables
    if (nhotvars>0)&&(iteration<iterations)
      J=(L==1 ? 1 : (l1>=1 ? layers1[1] : layers2[1]))
      for j=1:J
        i=1
        for hotvar=1:nhotvars
          imbalance=0
          hots=hotvars[hotvar]
          if hots==1
            i+=1
            break
          end
          ii=hots
          while ii>0
            imbalance+=w[1,j,i]*hotsums[i]
            ii-=1
            i+=1
          end
          imbalance=imbalance/nobs
          b[1,j]+=imbalance
          for ii=1:hots
            w[1,j,i-ii]=max(-c[1],min(c[1],w[1,j,i-ii]-(hotvals[i-ii]==0 ? 0 : imbalance/hotvals[i-ii])))
          end
        end
      end
    end
    push!(llv,h)
    if (mod(iteration,100)==0)||(iteration==iterations)  
      print("\e[2K")
      print("\e[1G")
      print("iteration ",iteration,"/",iterations,", log-likelihood=",h)
    end
  end
  println()
  return(w,b,llv)
end

function fitTimDist(w,b,data,events,layers1,layers2,iterations;hotvars=Int64[],obsweights=false,α=0.001,β1=0.9,β2=0.999,ϵ=1e-8,c=[3.0])
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  nevents=length(events)
  fevents=zeros(nevents)
  N,nvars=size(data,1),size(data,2)-(obsweights ? 1 : 0)-1
  sw2,sw3,sb2=size(w,2),size(w,3),size(b,2)
  x,u,∂h∂x,s1=ntuple(_->zeros(2,L,sb2),4)
  z,v,∂h∂z,s2=ntuple(_->zeros(L,sb2),4)
  cmax=length(c)
  llv=Float64[]
  # sum of observations per categorical value
  nhotvars=length(hotvars)
  if nhotvars>0
    hotsums=sum(data[:,1:sum(hotvars)].*(obsweights ? data[:,nvars+2] : ones(N)),dims=1)
    hotvals=minimum(data[:,1:sum(hotvars)],dims=1)+maximum(data[:,1:sum(hotvars)],dims=1)
    nobs=(obsweights ? sum(data[:,nvars+2]) : N)
  end
  # Adam optimization
  mb,vb=ntuple(_->zeros(L,sb2),2)
  mw,vw=ntuple(_->zeros(L,sw2,sw3),2)
  println()
  println("Maximizing likelihood time-to-event distribution..")
  for iteration=0:iterations
    ∂h∂w,∂h∂b=zeros(L,sw2,sw3),zeros(L,sb2)
    h=0
    for n=1:N
      obs=data[n,:]
      obsw=(obsweights ? obs[nvars+2] : 1)
      eventnr=nevents+1
      sumf=0
      if obs[nvars]==obs[nvars+1] # no censoring
        # propagate forward
        for l=1:L
          J=(l==1 ? nvars-((l1==0) ? 1 : 2) : (l<=l1+1 ? (layers1[l-1]+(l==l1+1 ? 1 : 0)) : (layers2[l-1-l1]+(l==l1+2 ? nevents : 0))))
          I=(l==L ? nevents+((l2==0) ? nevents : 0) : (l<=l1 ? layers1[l] : layers2[l-l1]+(l==l1+1 ? nevents : 0)))
          for i=1:I
            u[1,l,i]=b[l,i]
            v[l,i]=0
            for j=(l==l1+2 ? nevents+1 : 1):J
              u[1,l,i]+=w[l,i,j]*(((l==1)&(j<nvars-1)) ? obs[j] : (((l==l1+1)&&(j==J)) ? (i>nevents ? obs[nvars] : 0) : x[1,l-1,j]))
              if l>=l1+1
                v[l,i]+=w[l,i,j]*(l==l1+1 ? (((j==J)&&(i>nevents)) ? 1 : 0) : z[l-1,j])
              end
            end
            x[1,l,i]=((l==L)||((l==l1+1)&&(i<=nevents)) ? 1/(1+exp(-u[1,l,i])) : tanh(u[1,l,i]))
            s1[1,l,i]=((l==L)||((l==l1+1)&&(i<=nevents)) ? x[1,l,i]*(1-x[1,l,i]) : 1-x[1,l,i]^2)
            s2[l,i]=((l==L)||((l==l1+1)&&(i<=nevents)) ? (1-2*x[1,l,i])*s1[1,l,i] : -2*x[1,l,i]*s1[1,l,i])
            z[l,i]=((l<=l1)||((l==l1+1)&&(i<=nevents)) ? 0 : max(s1[1,l,i]*v[l,i],1e-18))
            if (l==l1+1)&&(i<=nevents)
              if obs[nvars-1]==events[i]
                eventnr=i
              end
              ∂h∂x[1,l,i]=obsw*(i<eventnr ? 1/(x[1,l,i]-1) : (i==eventnr ? 1/x[1,l,i] : 0))
              ∂h∂z[l,i]=0
              h+=obsw*(i<eventnr ? log(1-x[1,l,i]) : (i==eventnr ? log(x[1,l,i]) : 0))
            elseif (l==L)
              ∂h∂x[1,l,i]=0
              ∂h∂z[l,i]=obsw*(i==eventnr+((l2==0) ? nevents : 0) ? 1/z[l,i] : 0)
              h+=obsw*(i==eventnr+((l2==0) ? nevents : 0) ? log(z[l,i]) : 0)
            end
          end
        end
        # propagate backward
        if iteration<iterations
          l=L-1
          while l>=0
            I=(l==0 ? nvars-((l1==0) ? 1 : 2) : (l<=l1 ? (layers1[l]+(l==l1 ? 1 : 0)) : layers2[l-l1]+(l-1==l1 ? nevents : 0)))
            J=(l+1==L ? nevents+((l2==0) ? nevents : 0) : (l+1<=l1 ? layers1[l+1] : layers2[l+1-l1]+(l==l1 ? nevents : 0)))
            for i=1:I
              if (l>0)&&((i<I)||(l!=l1))&&((i>nevents)||(l!=l1+1))
                ∂h∂x[1,l,i]=0
                ∂h∂z[l,i]=0
              end
              for j=1:J
                if (l>0)&&((i<I)||(l!=l1))&&((i>nevents)||(l!=l1+1))
                  ∂h∂x[1,l,i]+=w[l+1,j,i]*(∂h∂x[1,l+1,j]*s1[1,l+1,j]+∂h∂z[l+1,j]*s2[l+1,j]*v[l+1,j])
                  ∂h∂z[l,i]+=w[l+1,j,i]*∂h∂z[l+1,j]*s1[1,l+1,j]
                end
                if i==1
                  ∂h∂b[l+1,j]+=∂h∂x[1,l+1,j]*s1[1,l+1,j]+∂h∂z[l+1,j]*s2[l+1,j]*v[l+1,j]
                  if n==N
                    mb[l+1,j]=β1*mb[l+1,j]+(1-β1)*∂h∂b[l+1,j]
                    vb[l+1,j]=β2*vb[l+1,j]+(1-β2)*∂h∂b[l+1,j]^2
                    b[l+1,j]+=α*mb[l+1,j]/((1-β1^(iteration+1))*(sqrt(vb[l+1,j]/(1-β2^(iteration+1)))+ϵ))
                  end
                end
                if !(((l==l1)&&(i==I)&&(j<=nevents))||((l==l1+1)&&(i<=nevents)))
                  ∂h∂w[l+1,j,i]+=∂h∂x[1,l+1,j]*s1[1,l+1,j]*(((l==0)&&(i<nvars-1)) ? obs[i] : (((l==l1)&&(i==I)) ? obs[nvars] : x[1,l,i]))+∂h∂z[l+1,j]*
                  (s2[l+1,j]*v[l+1,j]*(((l==0)&&(i<nvars-1)) ? obs[i] : (((l==l1)&&(i==I)) ? obs[nvars] : x[1,l,i]))+s1[1,l+1,j]*(((l==l1)&&(i==I)) ? 1 : (l==0 ? 0 : z[l,i])))
                  if n==N
                    mw[l+1,j,i]=β1*mw[l+1,j,i]+(1-β1)*∂h∂w[l+1,j,i]
                    vw[l+1,j,i]=β2*vw[l+1,j,i]+(1-β2)*∂h∂w[l+1,j,i]^2
                    w[l+1,j,i]=max(-c[min(2,cmax)],min(c[min(2,cmax)],w[l+1,j,i]+α*mw[l+1,j,i]/((1-β1^(iteration+1))*(sqrt(vw[l+1,j,i]/(1-β2^(iteration+1)))+ϵ))))
                    if l>l1 || ((l==l1)&&(i==I))
                      w[l+1,j,i]=max(w[l+1,j,i],0)
                    end
                  end
                end  
              end
            end
            l-=1
          end
        end
      else # censoring
        for t=1:2
          # propagate forward
          for l=1:L
            J=(l==1 ? nvars-((l1==0) ? 1 : 2) : (l<=l1+1 ? (layers1[l-1]+(l==l1+1 ? 1 : 0)) : (layers2[l-1-l1]+(l==l1+2 ? nevents : 0))))
            I=(l==L ? nevents+((l2==0) ? nevents : 0) : (l<=l1 ? layers1[l] : layers2[l-l1]+(l==l1+1 ? nevents : 0)))
            for i=1:I
              u[t,l,i]=b[l,i]
              for j=(l==l1+2 ? nevents+1 : 1):J
                u[t,l,i]+=w[l,i,j]*(((l==1)&&(j<nvars-1)) ? obs[j] : (((l==l1+1)&&(j==J)) ? (i>nevents ? obs[nvars+t-1] : 0) : x[t,l-1,j]))
              end
              x[t,l,i]=((l==L)||((l==l1+1)&&(i<=nevents)) ? 1/(1+exp(-u[t,l,i])) : tanh(u[t,l,i]))
              s1[t,l,i]=((l==L)||((l==l1+1)&&(i<=nevents)) ? x[t,l,i]*(1-x[t,l,i]) : 1-x[t,l,i]^2)
              if (t==2)
                if (l==l1+1)&&(i<=nevents)
                  fevents[i]=(i==1 ? 1 : fevents[i-1]*(1-x[t,l,i-1]))
                  if obs[nvars-1]==events[i]
                    eventnr=i
                  end
                  if (i==nevents)&&(eventnr>nevents)
                    sumf=fevents[i]*(1-x[1,l,i])
                  end
                elseif (l==L)&&((i-(l2==0 ? nevents : 0)==eventnr)||(eventnr>nevents))
                  sumf+=fevents[i-(l2==0 ? nevents : 0)]*x[t,l1+1,i-(l2==0 ? nevents : 0)]*(x[2,l,i]-x[1,l,i])
                end
              end
            end
          end
        end
        sumf=max(sumf,1e-18)
        h+=obsw*log(sumf)
        for i=1:nevents
          if eventnr>nevents
            ∂h∂x[1,l1+1,i]=-fevents[nevents]*(1-x[1,l1+1,nevents])/(1-x[1,l1+1,i])
          else
            ∂h∂x[1,l1+1,i]=0
          end
          if (i==eventnr)||(eventnr>nevents)
            ∂h∂x[1,l1+1,i]+=fevents[i]*(x[2,L,i+(l2==0 ? nevents : 0)]-x[1,L,i+(l2==0 ? nevents : 0)])
          end
          j=i
          while j<nevents
            j+=1
            if (j==eventnr)||(eventnr>nevents)
              ∂h∂x[1,l1+1,i]-=fevents[j]*x[1,l1+1,j]/(1-x[1,l1+1,i])*(x[2,L,j+(l2==0 ? nevents : 0)]-x[1,L,j+(l2==0 ? nevents : 0)])
            end
          end  
          ∂h∂x[1,l1+1,i]=obsw*∂h∂x[1,l1+1,i]/sumf
          if (i==eventnr)||(eventnr>nevents)
            ∂h∂x[1,L,i+(l2==0 ? nevents : 0)]=-obsw*fevents[i]*x[1,l1+1,i]/sumf
          else
            ∂h∂x[1,L,i+(l2==0 ? nevents : 0)]=0
          end
          ∂h∂x[2,l1+1,i]=0
          ∂h∂x[2,L,i+(l2==0 ? nevents : 0)]=-∂h∂x[1,L,i+(l2==0 ? nevents : 0)]
        end
        # propagate backward
        if iteration<iterations
          for t=1:2
            l=L-1
            while l>=0
              I=(l==0 ? nvars-((l1==0) ? 1 : 2) : (l<=l1 ? (layers1[l]+(l==l1 ? 1 : 0)) : layers2[l-l1]+(l-1==l1 ? nevents : 0)))
              J=(l+1==L ? nevents+((l2==0) ? nevents : 0) : (l+1<=l1 ? layers1[l+1] : layers2[l+1-l1]+(l==l1 ? nevents : 0)))
              for i=1:I
                if (l>0)&&((i<I)||(l!=l1))&&((i>nevents)||(l!=l1+1))
                  ∂h∂x[t,l,i]=0
                end
                for j=1:J
                  if (l>0)&&((i<I)||(l!=l1))&&((i>nevents)||(l!=l1+1))
                    ∂h∂x[t,l,i]+=w[l+1,j,i]*∂h∂x[t,l+1,j]*s1[t,l+1,j]
                  end
                  if i==1
                    ∂h∂b[l+1,j]+=∂h∂x[t,l+1,j]*s1[t,l+1,j]
                    if (n==N)&&(t==2)
                      mb[l+1,j]=β1*mb[l+1,j]+(1-β1)*∂h∂b[l+1,j]
                      vb[l+1,j]=β2*vb[l+1,j]+(1-β2)*∂h∂b[l+1,j]^2
                      b[l+1,j]+=α*mb[l+1,j]/((1-β1^(iteration+1))*(sqrt(vb[l+1,j]/(1-β2^(iteration+1)))+ϵ))
                    end
                  end
                  if !(((l==l1)&&(i==I)&&(j<=nevents))||((l==l1+1)&&(i<=nevents)))
                    ∂h∂w[l+1,j,i]+=∂h∂x[t,l+1,j]*s1[t,l+1,j]*(((l==0)&&(i<nvars-1)) ? obs[i] : (((l==l1)&&(i==I)) ? obs[nvars+t-1] : x[t,l,i]))
                    if (n==N)&&(t==2)
                      mw[l+1,j,i]=β1*mw[l+1,j,i]+(1-β1)*∂h∂w[l+1,j,i]
                      vw[l+1,j,i]=β2*vw[l+1,j,i]+(1-β2)*∂h∂w[l+1,j,i]^2
                      w[l+1,j,i]=max(-c[min(2,cmax)],min(c[min(2,cmax)],w[l+1,j,i]+α*mw[l+1,j,i]/((1-β1^(iteration+1))*(sqrt(vw[l+1,j,i]/(1-β2^(iteration+1)))+ϵ))))
                      if l>l1 || ((l==l1)&&(i==I))
                        w[l+1,j,i]=max(w[l+1,j,i],0)
                      end
                    end
                  end  
                end
              end
              l-=1
            end
          end  
        end
      end
    end  
    # balance hot-encoded categorical variables
    if (nhotvars>0)&&(iteration<iterations)
      J=(L==1 ? 2*nevents : (l1>=1 ? layers1[1] : layers2[1]+nevents))
      for j=1:J
        i=1
        for hotvar=1:nhotvars
          imbalance=0
          hots=hotvars[hotvar]
          if hots==1
            i+=1
            break
          end
          ii=hots
          while ii>0
            imbalance+=w[1,j,i]*hotsums[i]
            ii-=1
            i+=1
          end
          imbalance=imbalance/nobs
          b[1,j]+=imbalance
          for ii=1:hots
            w[1,j,i-ii]=max(-c[1],min(c[1],w[1,j,i-ii]-(hotvals[i-ii]==0 ? 0 : imbalance/hotvals[i-ii])))
          end
        end
      end
    end
    push!(llv,h)
    if (mod(iteration,100)==0)||(iteration==iterations)  
      print("\e[2K")
      print("\e[1G")
      print("iteration ",iteration,"/",iterations,", log-likelihood=",h)
    end
  end
  println()
  return(w,b,llv)
end

function evalCatPDF(obs,ncats,layers,w,b)
  L=length(layers)+1
  nvars=length(obs)
  pcats=zeros(ncats)
  x,u=ntuple(_->zeros(L,size(b,2)),2)
  p=1.0
  for l=1:L
    J=(l==1 ? nvars-1 : layers[l-1])
    I=(l==L ? ncats-1 : layers[l])
    for i=1:I
      u[l,i]=b[l,i]
      for j=1:J
        u[l,i]+=w[l,i,j]*(l==1 ? obs[j] : x[l-1,j])
      end
      x[l,i]=(l==L ? 1/(1+exp(-u[l,i])) : tanh(u[l,i]))
      if l==L
        pcats[i]=p*x[l,i]
        p-=pcats[i]
      end
    end
  end
  pcats[ncats]=p
  return(pcats)
end

function evalConPDF(obs,layers1,layers2,w,b)
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  nvars=length(obs)
  x,z,u,v,s1=ntuple(_->zeros(L,size(b,2)),5)
  for l=1:L
    J = (l==1 ? nvars-((l1==0) ? 0 : 1) : (l<=l1+1 ? (layers1[l-1]+(l==l1+1 ? 1 : 0)) : layers2[l-1-l1]))
    I = (l==L ? 1 : (l<=l1 ? layers1[l] : layers2[l-l1]))
    for i=1:I
      u[l,i]=b[l,i]
      v[l,i]=0
      for j=1:J
        u[l,i]+=w[l,i,j]*(l==1 ? obs[j] : (((l==l1+1)&&(j==J)) ? obs[nvars] : x[l-1,j]))
        if l>=l1+1
          v[l,i]+=w[l,i,j]*(l==l1+1 ? (j==J ? 1 : 0) : z[l-1,j])
        end
      end
      x[l,i]=(l==L ? 1/(1+exp(-u[l,i])) : tanh(u[l,i]))
      s1[l,i]=(l==L ? x[l,i]*(1-x[l,i]) : 1-x[l,i]^2)
      z[l,i]=(l<=l1 ? 0 : s1[l,i]*v[l,i])
    end
  end
  return(z[L,1])
end

function evalTimPDF(obs,nevents,layers1,layers2,w,b)
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  pevents,devents=ntuple(_->zeros(nevents),2)
  nvars=length(obs)+1
  x,z,u,v,s1=ntuple(_->zeros(L,size(b,2)),5)
  p=1.0
  for l=1:L
    J=(l==1 ? nvars-((l1==0) ? 1 : 2) : (l<=l1+1 ? (layers1[l-1]+(l==l1+1 ? 1 : 0)) : (layers2[l-1-l1]+(l==l1+2 ? nevents : 0))))
    I=(l==L ? nevents+((l2==0) ? nevents : 0) : (l<=l1 ? layers1[l] : layers2[l-l1]+(l==l1+1 ? nevents : 0)))
    for i=1:I
      u[l,i]=b[l,i]
      v[l,i]=0
      for j=(l==l1+2 ? nevents+1 : 1):J
        u[l,i]+=w[l,i,j]*(((l==1)&(j<nvars-1)) ? obs[j] : (((l==l1+1)&&(j==J)) ? (i>nevents ? obs[nvars-1] : 0) : x[l-1,j]))
        if l>=l1+1
          v[l,i]+=w[l,i,j]*(l==l1+1 ? (((j==J)&&(i>nevents)) ? 1 : 0) : z[l-1,j])
        end
      end
      x[l,i]=((l==L)||((l==l1+1)&&(i<=nevents)) ? 1/(1+exp(-u[l,i])) : tanh(u[l,i]))
      s1[l,i]=((l==L)||((l==l1+1)&&(i<=nevents)) ? x[l,i]*(1-x[l,i]) : 1-x[l,i]^2)
      z[l,i]=((l<=l1)||((l==l1+1)&&(i<=nevents)) ? 0 : s1[l,i]*v[l,i])
      if (l==l1+1)&&(i<=nevents)
        pevents[i]=p*x[l,i]
        p-=pevents[i]
      end
      if (l==L)&&((l2>0)||(i>nevents))
        devents[i-((l2==0) ? nevents : 0)]=z[L,i]
      end
    end
  end
  return(pevents,devents)
end

function evalConCDF(obs,layers1,layers2,w,b)
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  nvars=length(obs)
  x,u=ntuple(_->zeros(L,size(b,2)),2)
  for l=1:L
    J = (l==1 ? nvars-((l1==0) ? 0 : 1) : (l<=l1+1 ? (layers1[l-1]+(l==l1+1 ? 1 : 0)) : layers2[l-1-l1]))
    I = (l==L ? 1 : (l<=l1 ? layers1[l] : layers2[l-l1]))
    for i=1:I
      u[l,i]=b[l,i]
      for j=1:J
        u[l,i]+=w[l,i,j]*(l==1 ? obs[j] : (((l==l1+1)&&(j==J)) ? obs[nvars] : x[l-1,j]))
      end
      x[l,i]=(l==L ? 1/(1+exp(-u[l,i])) : tanh(u[l,i]))
    end
  end
  return(x[L,1])
end

function evalTimCDF(obs,nevents,layers1,layers2,w,b)
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  pevents,devents=ntuple(_->zeros(nevents),2)
  nvars=length(obs)+1
  x,u=ntuple(_->zeros(L,size(b,2)),2)
  p=1.0
  for l=1:L
    J=(l==1 ? nvars-((l1==0) ? 1 : 2) : (l<=l1+1 ? (layers1[l-1]+(l==l1+1 ? 1 : 0)) : (layers2[l-1-l1]+(l==l1+2 ? nevents : 0))))
    I=(l==L ? nevents+((l2==0) ? nevents : 0) : (l<=l1 ? layers1[l] : layers2[l-l1]+(l==l1+1 ? nevents : 0)))
    for i=1:I
      u[l,i]=b[l,i]
      for j=(l==l1+2 ? nevents+1 : 1):J
        u[l,i]+=w[l,i,j]*(((l==1)&&(j<nvars-1)) ? obs[j] : (((l==l1+1)&&(j==J)) ? (i>nevents ? obs[nvars-1] : 0) : x[l-1,j]))
      end
      x[l,i]=((l==L)||((l==l1+1)&&(i<=nevents)) ? 1/(1+exp(-u[l,i])) : tanh(u[l,i]))
      if (l==l1+1)&&(i<=nevents)
        pevents[i]=p*x[l,i]
        p-=pevents[i]
      end
      if (l==L)&&((l2>0)||(i>nevents))
        devents[i-((l2==0) ? nevents : 0)]=x[L,i]
      end
    end
  end
  return(pevents,devents)
end

function limitsConCDF(layers1,layers2,w,b)
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  x,u=ntuple(_->zeros(L,size(b,2)),2)
  # calculate infimum
  for l=(l1+1):L
    J = (l==l1+1 ? 1 : layers2[l-1-l1])
    I = (l==L ? 1 : layers2[l-l1])
    for i=1:I
      u[l,i]=b[l,i]
      for j=1:J
        u[l,i]+=w[l,i,j]*((l==l1+1) ? 0 : x[l-1,j])
      end
      x[l,i]=(l==L ? ((l==l1+1) ? 0 : 1/(1+exp(-u[l,i]))) : ((l==l1+1) ? -1 : tanh(u[l,i])))
    end
  end
  inf=x[L,1]
  # calculate supremum
  for l=(l1+1):L
    J = (l==l1+1 ? 1 : layers2[l-1-l1])
    I = (l==L ? 1 : layers2[l-l1])
    for i=1:I
      u[l,i]=b[l,i]
      for j=1:J
        u[l,i]+=w[l,i,j]*((l==l1+1) ? 0 : x[l-1,j])
      end
      x[l,i]=(l==L ? ((l==l1+1) ? 1 : 1/(1+exp(-u[l,i]))) : ((l==l1+1) ? 1 : tanh(u[l,i])))
    end
  end
  sup=x[L,1]
  return(inf,sup)
end

function limitsTimCDF(nevents,layers1,layers2,w,b)
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  inf,sup=ntuple(_->zeros(nevents),2)
  x,u=ntuple(_->zeros(L,size(b,2)),2)
  # calculate infimum
  for l=(l1+1):L
    J=(l==l1+1 ? 1 : (layers2[l-1-l1]+(l==l1+2 ? nevents : 0)))
    I=(l==L ? nevents+((l2==0) ? nevents : 0) : layers2[l-l1]+(l==l1+1 ? nevents : 0))
    for i=(l==l1+1 ? nevents+1 : 1):I
      u[l,i]=b[l,i]
      for j=(l==l1+2 ? nevents+1 : 1):J
        u[l,i]+=w[l,i,j]*((l==l1+1) ? 0 : x[l-1,j])
      end
      x[l,i]=(l==L ? ((l==l1+1) ? 0 : 1/(1+exp(-u[l,i]))) : ((l==l1+1) ? -1 : tanh(u[l,i])))
      if (l==L)&&((l2>0)||(i>nevents))
        inf[i-((l2==0) ? nevents : 0)]=x[L,i]
      end
    end
  end
  # calculate supremum
  for l=(l1+1):L
    J=(l==l1+1 ? 1 : (layers2[l-1-l1]+(l==l1+2 ? nevents : 0)))
    I=(l==L ? nevents+((l2==0) ? nevents : 0) : layers2[l-l1]+(l==l1+1 ? nevents : 0))
    for i=(l==l1+1 ? nevents+1 : 1):I
      u[l,i]=b[l,i]
      for j=(l==l1+2 ? nevents+1 : 1):J
        u[l,i]+=w[l,i,j]*((l==l1+1) ? 0 : x[l-1,j])
      end
      x[l,i]=(l==L ? ((l==l1+1) ? 1 : 1/(1+exp(-u[l,i]))) : ((l==l1+1) ? 1 : tanh(u[l,i])))
      if (l==L)&&((l2>0)||(i>nevents))
        sup[i-((l2==0) ? nevents : 0)]=x[L,i]
      end
    end
  end
  return(inf,sup)
end

function quantileCon(cvars,p,layers1,layers2,w,b;accuracy=1e-8,boundLow=-1e6,boundUpp=1e6)
  inf,sup=limitsCDF(layers1,layers2,w,b)
  u=inf+(sup-inf)*p
  xLow,xUpp=-1,1
  yLow=evalCDF([cvars xLow],layers1,layers2,w,b)
  yUpp=evalCDF([cvars xUpp],layers1,layers2,w,b)
  while ((yLow>u)&&(xLow>boundLow))
    xUpp,yUpp=xLow,yLow
    xLow=2*xLow
    yLow=evalCDF([cvars xLow],layers1,layers2,w,b)
  end
  while ((yUpp<u)&&(xUpp<boundUpp))
    xLow,yLow=xUpp,yUpp
    xUpp=2*xUpp
    yUpp=evalCDF([cvars xUpp],layers1,layers2,w,b)
  end
  while ((xUpp-xLow)>accuracy)
    xInt=xLow+(xUpp-xLow)/2
    yInt=evalCDF([cvars xInt],layers1,layers2,w,b)
    if (yInt>u)
      xUpp,yUpp=xInt,yInt
    else
      xLow,yLow=xInt,yInt
    end
  end
  return(xLow+(xUpp-xLow)/2)
end

function quantileTim(cvars,eventnr,nevents,p,layers1,layers2,w,b;accuracy=1e-8,boundLow=-1e6,boundUpp=1e6)
  inf,sup=limitsTimCDF(nevents,layers1,layers2,w,b)
  u=inf[eventnr]+(sup[eventnr]-inf[eventnr])*p
  xLow,xUpp=-1,1
  yLow=evalTimCDF([cvars xLow],nevents,layers1,layers2,w,b)[2][eventnr]
  yUpp=evalTimCDF([cvars xUpp],nevents,layers1,layers2,w,b)[2][eventnr]
  while ((yLow>u)&&(xLow>boundLow))
    xUpp,yUpp=xLow,yLow
    xLow=2*xLow
    yLow=evalTimCDF([cvars xLow],nevents,layers1,layers2,w,b)[2][eventnr]
  end
  while ((yUpp<u)&&(xUpp<boundUpp))
    xLow,yLow=xUpp,yUpp
    xUpp=2*xUpp
    yUpp=evalTimCDF([cvars xUpp],nevents,layers1,layers2,w,b)[2][eventnr]
  end
  while ((xUpp-xLow)>accuracy)
    xInt=xLow+(xUpp-xLow)/2
    yInt=evalTimCDF([cvars xInt],nevents,layers1,layers2,w,b)[2][eventnr]
    if (yInt>u)
      xUpp,yUpp=xInt,yInt
    else
      xLow,yLow=xInt,yInt
    end
  end
  return(xLow+(xUpp-xLow)/2)
end
