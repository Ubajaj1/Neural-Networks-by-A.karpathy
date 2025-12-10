
###compare department salary with company salary

from numpy import long
with cte as(Select s.employee_id,s.amount, Date_format(s.pay_date,'%Y-%m') as pay_month, 
e.department_id, Avg(amount) over (partition by Date_format(s.pay_date,'%Y-%m') as pay_month)) as company_avg,
avg(amount) over partition by e.department_id, Date_format(s.pay_date,'%Y-%m') as pay_month)) as department_avg
from salary s
left join employee e
on s.employee_id = e.employee_id)


Select Distinct pay_month, department_id, 
case when department_avg > company_avg then 'higher'
when department_avg < company_avg then 'lower'
else 'same'
end as comparison
from cte

### policy holders and investments
# link: https://www.youtube.com/watch?v=BVGhF9yaDU0&list=WL&index=10&t=771s

with cte as(Select concat(lat,',',lon) as location
from insurance
group by lat,lon
having count(pid)>1),

cte2 as (Select distinct i1.* 
from insurance i1
left join insurance i2
on i1.tiv_2015=i2.tiv_2015
where i1.pid<>i2.pid
and concat(i1.lat,',',i1.lon) not in (select location from cte))

select round(sum(tiv_2016),2) as sum
from cte2







